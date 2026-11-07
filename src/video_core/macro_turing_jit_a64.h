// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <oaknut/code_block.hpp>
#include <oaknut/oaknut.hpp>

#include "video_core/macro_turing.h"

namespace Tegra {

// AArch64 port of macro_turing_jit.h (same State layout, same block structure, same semantics):
// every instruction of the uploaded macro memory compiled once, falling through to the next one,
// branches and loops through a jump table. Emits and register reads call back into C++.
class TuringMacroJit {
public:
    using Word = std::uint32_t;
    using Result = TuringMacroInterpreter::Result;

    struct State {
        std::array<Word, 24> regs;
        Word carry;
        Word method;
        Word increment;
        Word loop_count;
        Word loop_start;
        Word loop_end;
        Word next_pc;
        Word steps;
        Word start;
        Word consumed;
        Word params_size;
        Word fail_pc;
        std::uint8_t end_next;
        std::uint8_t stopped;
        std::uint8_t cancel_end;
        std::uint8_t pad;
        Word scratch_x;
        Word scratch_y;
        const Word* params;
        Word* dram;
        void* user;
        bool (*emit)(State*, Word method, Word value);
        Word (*read)(State*, Word address);
        const std::vector<Word>* fifo;
    };

    enum Error : Word {
        None = 0,
        FifoUnderflow,
        InvalidSource,
        InvalidDestination,
        UnsupportedExtended,
        InvalidMerge,
        InvalidLoop,
        DataRamReadOutOfBounds,
        DataRamWriteOutOfBounds,
        InvalidOutput,
        UnsupportedEmit,
        OutsideCode,
        StepLimit,
    };

    static std::string_view ErrorText(Word error) {
        constexpr std::array<std::string_view, 13> texts{
            "",
            "parameter FIFO underflow",
            "invalid source register",
            "invalid destination register",
            "unsupported extended operation",
            "invalid merge destination",
            "invalid loop",
            "data RAM read out of bounds",
            "data RAM write out of bounds",
            "invalid output selector",
            "unsupported emitted method",
            "instruction outside uploaded RAM",
            "instruction limit exceeded",
        };
        return error < texts.size() ? texts[error] : "unknown";
    }

    /// Compiles `code` (3 words per instruction) when it differs from what was compiled.
    /// Returns false when it could not be compiled (the caller interprets it instead).
    bool Prepare(std::span<const Word> code, std::uint64_t generation) {
        if (generation == compiled_generation) {
            return program != nullptr;
        }
        compiled_generation = generation;
        program.reset();
        // Conditional branches reach +-1 MiB; a program too large for that is generated again
        // with every far conditional branch as an inverted branch around a B.
#ifdef TURING_MACRO_JIT_TEST_FAR
        for (const bool far : {true}) {
#else
        for (const bool far : {false, true}) {
#endif
            try {
                program = std::make_unique<Program>(code, far);
                return true;
            } catch (...) {
                program.reset();
            }
        }
        return false;
    }

    template <typename Read, typename Emit>
    Result Execute(Word start, std::vector<Word>& parameters, std::span<Word> data_ram, Read read,
                   Emit emit) {
        State state;
        state.regs.fill(0);
        state.carry = state.method = state.increment = 0;
        state.loop_count = state.loop_start = state.loop_end = 0;
        state.consumed = 0;
        state.fail_pc = 0;
        state.end_next = state.stopped = state.cancel_end = 0;
        state.start = start;
        state.steps = 1000000;
        state.params = parameters.data();
        state.params_size = static_cast<Word>(parameters.size());
        state.dram = data_ram.data();
        state.fifo = &parameters;
        Thunks<Read, Emit> thunks{read, emit};
        state.user = &thunks;
        state.emit = &Thunks<Read, Emit>::CallEmit;
        state.read = &Thunks<Read, Emit>::CallRead;
        if (!program) {
            return {false, 0, 0, "macro code not compiled"};
        }
        const Word error = program->entry(&state);
        if (error == None) {
            return {true, state.fail_pc, state.consumed, {}};
        }
        return {false, state.fail_pc, state.consumed, ErrorText(error)};
    }

private:
    template <typename Read, typename Emit>
    struct Thunks {
        Read& read;
        Emit& emit;
        static bool CallEmit(State* state, Word method, Word value) {
            const bool ok = static_cast<Thunks*>(state->user)->emit(method, value);
            // MME_DMA_READ_FIFOED appends to the FIFO from inside the emit.
            state->params = state->fifo->data();
            state->params_size = static_cast<Word>(state->fifo->size());
            return ok;
        }
        static Word CallRead(State* state, Word address) {
            return static_cast<Thunks*>(state->user)->read(address);
        }
    };

    struct Program {
        Program(std::span<const Word> code, bool far_branches) : far{far_branches} {
            const Word count = static_cast<Word>(code.size() / 3);
            table.resize(count);
            std::vector<std::uint32_t> words;
            words.reserve(std::size_t{count} * 64 + 256);
            oaknut::VectorCodeGenerator gen{words};
            std::vector<std::ptrdiff_t> offsets(count);
            Generate(gen, code, offsets);
            block = std::make_unique<oaknut::CodeBlock>((words.size() * 4 + 0xfff) & ~std::size_t{0xfff});
            block->unprotect();
            std::memcpy(block->ptr(), words.data(), words.size() * 4);
            block->protect();
            block->invalidate_all();
            const auto base = reinterpret_cast<const std::uint8_t*>(block->ptr());
            for (Word pc = 0; pc < count; ++pc) {
                table[pc] = base + offsets[pc];
            }
            entry = reinterpret_cast<Word (*)(State*)>(block->ptr());
        }

        Word (*entry)(State*) = nullptr;

    private:
        using Decoded = TuringMacroInterpreter::Decoded;
        using Gen = oaknut::VectorCodeGenerator;
        using WReg = oaknut::WReg;
        using Label = oaknut::Label;
        using Cond = oaknut::Cond;

        // X19: State*; W20/W21: the instruction's FIFO loads; W22/W23: its ALU results;
        // X24: the jump table. W0/W1/W2 play eax/ecx/edx of the x86-64 version, W9/W10 scratch.
        static constexpr WReg LOAD0{20}, LOAD1{21}, ALU0{22}, ALU1{23};

        static constexpr std::size_t RegOffset(Word reg) {
            return offsetof(State, regs) + reg * sizeof(Word);
        }
        static constexpr Cond Invert(Cond cond) {
            return static_cast<Cond>(static_cast<int>(cond) ^ 1);
        }

        // Branches to the shared tail (dispatch, success, failures) can be far.
        void BCond(Cond cond, Label& target) {
            if (!far) {
                g->B(cond, target);
                return;
            }
            Label skip;
            g->B(Invert(cond), skip);
            g->B(target);
            g->l(skip);
        }
        void Cbz(WReg reg, Label& target) {
            if (!far) {
                g->CBZ(reg, target);
                return;
            }
            Label skip;
            g->CBNZ(reg, skip);
            g->B(target);
            g->l(skip);
        }
        void Cbnz(WReg reg, Label& target) {
            if (!far) {
                g->CBNZ(reg, target);
                return;
            }
            Label skip;
            g->CBZ(reg, skip);
            g->B(target);
            g->l(skip);
        }

        void LoadState(WReg dst, std::size_t offset) {
            g->LDR(dst, oaknut::util::X19, offset);
        }
        void StoreState(WReg src, std::size_t offset) {
            g->STR(src, oaknut::util::X19, offset);
        }
        void StoreStateImm(Word value, std::size_t offset) {
            using namespace oaknut::util;
            if (value == 0) {
                g->STR(WZR, X19, offset);
                return;
            }
            g->MOV(W9, value);
            g->STR(W9, X19, offset);
        }
        void StoreStateByte(std::uint8_t value, std::size_t offset) {
            using namespace oaknut::util;
            if (value == 0) {
                g->STRB(WZR, X19, offset);
                return;
            }
            g->MOV(W9, Word{value});
            g->STRB(W9, X19, offset);
        }

        void Generate(Gen& gen, std::span<const Word> code, std::vector<std::ptrdiff_t>& offsets) {
            using namespace oaknut::util;
            g = &gen;
            const Word count = static_cast<Word>(code.size() / 3);
            labels.resize(count);
            std::vector<Decoded> decoded(count);
            for (Word pc = 0; pc < count; ++pc) {
                decoded[pc] = TuringMacroInterpreter::Decode(
                    {code[pc * 3 + 2], code[pc * 3 + 1], code[pc * 3]});
            }
            Analyze(decoded);
            g->STP(X29, X30, SP, PRE_INDEXED, -64);
            g->STP(X19, X20, SP, 16);
            g->STP(X21, X22, SP, 32);
            g->STP(X23, X24, SP, 48);
            g->MOV(X19, X0);
            g->MOV(X24, reinterpret_cast<std::uint64_t>(table.data()));
            LoadState(W0, offsetof(State, start));
            g->MOV(W10, count);
            g->CMP(W0, W10);
            BCond(HS, fail_outside);
            g->LDR(X9, X24, W0, UXTW, 3);
            g->BR(X9);

            for (Word pc = 0; pc < count; ++pc) {
                offsets[pc] = g->offset();
                g->l(labels[pc]);
                Instruction(pc, count, decoded[pc]);
            }

            // Dispatch on W0 (the next pc). Every backward edge comes through here, so the
            // instruction limit is counted here.
            g->l(dispatch);
            LoadState(W9, offsetof(State, steps));
            g->SUBS(W9, W9, 1);
            StoreState(W9, offsetof(State, steps));
            g->B(EQ, fail_steps);
            g->MOV(W10, count);
            g->CMP(W0, W10);
            g->B(HS, fail_outside);
            g->LDR(X9, X24, W0, UXTW, 3);
            g->BR(X9);

            g->l(success);
            g->MOV(W0, WZR);
            g->l(leave);
            g->LDP(X23, X24, SP, 48);
            g->LDP(X21, X22, SP, 32);
            g->LDP(X19, X20, SP, 16);
            g->LDP(X29, X30, SP, POST_INDEXED, 64);
            g->RET();
            const auto fail_with = [&](Label& label, Word error) {
                g->l(label);
                g->MOV(W0, error);
                g->B(leave);
            };
            fail_with(fail_underflow, FifoUnderflow);
            fail_with(fail_source, InvalidSource);
            fail_with(fail_destination, InvalidDestination);
            fail_with(fail_extended, UnsupportedExtended);
            fail_with(fail_merge, InvalidMerge);
            fail_with(fail_loop, InvalidLoop);
            fail_with(fail_dread, DataRamReadOutOfBounds);
            fail_with(fail_dwrite, DataRamWriteOutOfBounds);
            fail_with(fail_output, InvalidOutput);
            fail_with(fail_emit, UnsupportedEmit);
            fail_with(fail_outside, OutsideCode);
            fail_with(fail_steps, StepLimit);
        }

        // Loads MME register `reg` (as an operand of `lane`) into `dst`.
        void Get(WReg dst, const Decoded& d, Word reg, Word lane) {
            using namespace oaknut::util;
            if (reg < 24) {
                LoadState(dst, RegOffset(reg));
                return;
            }
            switch (reg) {
            case 24: g->MOV(dst, WZR); return;
            case 25: g->MOV(dst, Word(std::int32_t(std::int16_t(d.imm[lane])))); return;
            case 26: g->MOV(dst, Word(std::int32_t(std::int16_t(d.imm[1 - lane])))); return;
            case 27: g->MOV(dst, (d.imm[0] << 16) | d.imm[1]); return;
            case 28: g->MOV(dst, LOAD0); return;
            case 29: g->MOV(dst, LOAD1); return;
            default: g->B(fail_source); return;
            }
        }

        static bool IsFlow(Word op) {
            return op >= 23 && op <= 29;
        }

        // Where a loop can end (its back edge is checked only there) and which instructions can
        // run right after an END (only those check for the end of the macro).
        void Analyze(const std::vector<Decoded>& decoded) {
            const Word count = static_cast<Word>(decoded.size());
            loop_end.assign(count, false);
            after_end.assign(count, false);
            const auto mark = [&](std::vector<bool>& set, std::int64_t pc) {
                if (pc >= 0 && pc < count) set[static_cast<std::size_t>(pc)] = true;
            };
            for (Word pc = 0; pc < count; ++pc) {
                for (Word lane = 0; lane < 2; ++lane) {
                    const Decoded& d = decoded[pc];
                    if (d.op[lane] == 23 && d.imm[lane] >= 2 && pc + d.imm[lane] - 1 < count) {
                        loop_end[pc + d.imm[lane] - 1] = true;
                    }
                }
            }
            for (Word pc = 0; pc < count; ++pc) {
                const Decoded& d = decoded[pc];
                if (!d.end) continue;
                mark(after_end, std::int64_t{pc} + 1);
                for (Word lane = 0; lane < 2; ++lane) {
                    const Word imm = d.imm[lane];
                    switch (d.op[lane]) {
                    case 23: mark(after_end, std::int64_t{pc} + imm); break;
                    case 24: mark(after_end, std::int64_t{pc} + (imm & 0x7fff)); break;
                    case 25: case 26: case 27: case 28: case 29:
                        mark(after_end, std::int64_t{pc} + (std::int32_t(imm << 19) >> 19));
                        break;
                    default: break;
                    }
                }
            }
            // An END at a loop's last instruction can be followed by the loop's second one.
            for (Word pc = 0; pc < count; ++pc) {
                for (Word lane = 0; lane < 2; ++lane) {
                    const Decoded& d = decoded[pc];
                    if (d.op[lane] == 23 && d.imm[lane] >= 2) {
                        const Word end = pc + d.imm[lane] - 1;
                        if (end < count && decoded[end].end) mark(after_end, std::int64_t{pc} + 1);
                    }
                }
            }
        }

        void Instruction(Word pc, Word count, const Decoded& d) {
            using namespace oaknut::util;
            // FIFO loads, in lane order.
            for (Word i = 0; i < 2; ++i) {
                const WReg load = i == 0 ? LOAD0 : LOAD1;
                if (!d.load[i]) {
                    g->MOV(load, WZR);
                    continue;
                }
                LoadState(W0, offsetof(State, consumed));
                LoadState(W9, offsetof(State, params_size));
                g->CMP(W0, W9);
                BCond(HS, fail_underflow);
                g->LDR(X10, X19, offsetof(State, params));
                g->LDR(load, X10, W0, UXTW, 2);
                g->ADD(W0, W0, 1);
                StoreState(W0, offsetof(State, consumed));
            }
            if (!d.pred_mode) {
                Body(pc, count, d, 15, true);
                return;
            }
            Label when_false;
            Get(W0, d, d.pred_reg, 0);
            if (far) {
                // The true body can be long enough to need the far form too.
                Label when_true;
                g->CBNZ(W0, when_true);
                g->B(when_false);
                g->l(when_true);
            } else {
                g->CBZ(W0, when_false);
            }
            Body(pc, count, d, d.mask_true, false);
            g->l(when_false);
            Body(pc, count, d, d.mask_false, true);
        }

        // `falls_to_next`: the code of instruction pc + 1 follows this body directly.
        void Body(Word pc, Word count, const Decoded& d, Word mask, bool falls_to_next) {
            using namespace oaknut::util;
            // An inactive lane's result reads as 0 when an output selects it.
            for (Word lane = 0; lane < 2; ++lane) {
                if (mask & (1U << lane)) continue;
                bool selected = false;
                for (Word out = 0; out < 2; ++out) {
                    if (!(mask & (4U << out))) continue;
                    selected |= d.out_method[out] == lane + 1 || d.out_emit[out] == lane + 1;
                }
                if (selected) {
                    g->MOV(lane == 0 ? ALU0 : ALU1, WZR);
                }
            }
            const bool flow = ((mask & 1) && IsFlow(d.op[0])) || ((mask & 2) && IsFlow(d.op[1]));
            if (flow) {
                StoreStateImm(pc + 1, offsetof(State, next_pc));
                StoreStateByte(0, offsetof(State, cancel_end));
            }
            // Lane 1's operands are read before lane 0 commits its destination (or calls out).
            const bool lane0_active = (mask & 1) != 0;
            const bool early_operands =
                (mask & 2) && lane0_active &&
                (d.op[0] == 22 || (d.dst[0] < 24 && (d.dst[0] == d.src_a[1] || d.dst[0] == d.src_b[1])));
            if (early_operands) {
                Get(W0, d, d.src_a[1], 1);
                StoreState(W0, offsetof(State, scratch_x));
                Get(W0, d, d.src_b[1], 1);
                StoreState(W0, offsetof(State, scratch_y));
            }
            for (Word lane = 0; lane < 2; ++lane) {
                if (!(mask & (1U << lane))) {
                    continue;
                }
                if (lane == 1 && early_operands) {
                    LoadState(W1, offsetof(State, scratch_x));
                    LoadState(W2, offsetof(State, scratch_y));
                } else {
                    Get(W1, d, d.src_a[lane], lane);
                    Get(W2, d, d.src_b[lane], lane);
                }
                Lane(pc, d, lane);
                // W0 holds the lane's value.
                g->MOV(lane == 0 ? ALU0 : ALU1, W0);
                if (d.dst[lane] < 24) {
                    StoreState(W0, RegOffset(d.dst[lane]));
                } else if (d.dst[lane] != 24) {
                    g->B(fail_destination);
                }
            }
            for (Word lane = 0; lane < 2; ++lane) {
                if (!(mask & (4U << lane))) {
                    continue;
                }
                if (d.out_method[lane]) {
                    Output(W0, d, d.out_method[lane]);
                    g->AND(W1, W0, 0xfff);
                    StoreState(W1, offsetof(State, method));
                    g->UBFX(W0, W0, 12, 4);
                    StoreState(W0, offsetof(State, increment));
                }
                if (d.out_emit[lane]) {
                    Output(W2, d, d.out_emit[lane]);
                    LoadState(W1, offsetof(State, method));
                    g->MOV(X0, X19);
                    g->LDR(X9, X19, offsetof(State, emit));
                    g->BLR(X9);
                    // bool in the low byte of W0 (the rest is unspecified).
                    g->TST(W0, 0xff);
                    BCond(EQ, fail_emit);
                    LoadState(W0, offsetof(State, increment));
                    LoadState(W1, offsetof(State, method));
                    g->ADD(W1, W1, W0);
                    StoreState(W1, offsetof(State, method));
                }
            }
            // An END in the previous instruction ends the macro after this one.
            if (after_end[pc]) {
                g->LDRB(W9, X19, offsetof(State, end_next));
                Cbnz(W9, success);
            }
            if (flow) {
                g->LDRB(W9, X19, offsetof(State, stopped));
                Cbnz(W9, success);
            }
            if (d.end) {
                if (flow) {
                    g->LDRB(W9, X19, offsetof(State, cancel_end));
                    g->EOR(W9, W9, 1);
                    g->STRB(W9, X19, offsetof(State, end_next));
                } else {
                    StoreStateByte(1, offsetof(State, end_next));
                }
            }
            if (loop_end[pc]) {
                Label no_loop;
                LoadState(W0, offsetof(State, loop_count));
                g->CBZ(W0, no_loop);
                LoadState(W9, offsetof(State, loop_end));
                g->MOV(W10, pc);
                g->CMP(W9, W10);
                g->B(NE, no_loop);
                g->SUB(W0, W0, 1);
                StoreState(W0, offsetof(State, loop_count));
                LoadState(W0, offsetof(State, loop_start));
                g->ADD(W0, W0, 1);
                g->B(dispatch);
                g->l(no_loop);
            }
            if (flow) {
                LoadState(W0, offsetof(State, next_pc));
                if (pc + 1 < count) {
                    g->MOV(W10, pc + 1);
                    g->CMP(W0, W10);
                    BCond(NE, dispatch);
                    if (!falls_to_next) {
                        g->B(labels[pc + 1]);
                    }
                } else {
                    g->B(dispatch);
                }
            } else if (pc + 1 < count) {
                if (!falls_to_next) {
                    g->B(labels[pc + 1]);
                }
            } else {
                g->B(fail_outside);
            }
        }

        void Output(WReg dst, const Decoded& d, Word selector) {
            switch (selector) {
            case 1: g->MOV(dst, ALU0); return;
            case 2: g->MOV(dst, ALU1); return;
            case 3: g->MOV(dst, LOAD0); return;
            case 4: g->MOV(dst, LOAD1); return;
            case 5: g->MOV(dst, d.imm[0]); return;
            case 6: g->MOV(dst, d.imm[1]); return;
            case 8: g->MOV(dst, d.imm[0] >> 12); return;
            case 9: g->MOV(dst, d.imm[1] >> 12); return;
            case 10: g->MOV(dst, (d.imm[0] << 16) | d.imm[1]); return;
            default: g->B(fail_output); return;
            }
        }

        // The condition of comparison `op` after CMP W1, W2.
        static Cond Condition(Word op) {
            switch (op) {
            case 17: case 25: return Cond::LT;
            case 18: case 26: return Cond::LO;
            case 19: case 27: return Cond::LE;
            case 20: case 28: return Cond::LS;
            default: return Cond::EQ;
            }
        }

        // W1 = x, W2 = y; leaves the value in W0.
        void Lane(Word pc, const Decoded& d, Word lane) {
            using namespace oaknut::util;
            const Word imm = d.imm[lane];
            const Word op = d.op[lane];
            switch (op) {
            case 0:
                g->ADDS(W0, W1, W2);
                g->CSET(W9, CS);
                StoreState(W9, offsetof(State, carry));
                return;
            case 1:
                g->ADD(W0, W1, W2);
                LoadState(W9, offsetof(State, carry));
                g->ADD(W0, W0, W9);
                return;
            case 2:
                // The carry is the borrow: x < y.
                g->SUBS(W0, W1, W2);
                g->CSET(W9, LO);
                StoreState(W9, offsetof(State, carry));
                return;
            case 3:
                g->SUB(W0, W1, W2);
                LoadState(W9, offsetof(State, carry));
                g->SUB(W0, W0, W9);
                return;
            case 4:
            case 6:
                if (op == 4) {
                    g->SMULL(X0, W1, W2);
                } else {
                    g->UMULL(X0, W1, W2);
                }
                g->LSR(X9, X0, 32);
                StoreState(W9, offsetof(State, carry));
                g->MOV(W0, W0);
                return;
            case 5:
                LoadState(W0, offsetof(State, carry));
                return;
            case 7:
                g->MOV(W9, 0x1000);
                g->CMP(W1, W9);
                BCond(NE, fail_extended);
                g->MOV(W0, WZR);
                return;
            case 8: g->CLZ(W0, W1); return;
            case 9: g->LSLV(W0, W1, W2); return;
            case 10: g->LSRV(W0, W1, W2); return;
            case 11: g->ASRV(W0, W1, W2); return;
            case 12: g->AND(W0, W1, W2); return;
            case 13: g->AND(W0, W1, W2); g->MVN(W0, W0); return;
            case 14: g->ORR(W0, W1, W2); return;
            case 15: g->EOR(W0, W1, W2); return;
            case 16: {
                const Word dest = (imm >> 10) & 63, bit_count = (imm >> 5) & 31;
                const Word source = imm & 31;
                if (dest >= 32) {
                    g->B(fail_merge);
                    g->MOV(W0, WZR);
                    return;
                }
                const Word bits_mask = (1U << bit_count) - 1;
                g->MOV(W9, Word(~(bits_mask << dest)));
                g->AND(W0, W1, W9);
                if (source != 0) {
                    g->LSR(W2, W2, source);
                }
                g->MOV(W9, bits_mask);
                g->AND(W2, W2, W9);
                if (dest != 0) {
                    g->LSL(W2, W2, dest);
                }
                g->ORR(W0, W0, W2);
                return;
            }
            case 17: case 18: case 19: case 20: case 21:
                g->CMP(W1, W2);
                g->CSETM(W0, Condition(op));
                return;
            case 22:
                g->ADD(W1, W1, W2);
                g->AND(W1, W1, 0xffff);
                g->MOV(X0, X19);
                g->LDR(X9, X19, offsetof(State, read));
                g->BLR(X9);
                g->MOV(W0, W0);
                return;
            case 23: {
                if (imm < 2) {
                    g->B(fail_loop);
                    g->MOV(W0, WZR);
                    return;
                }
                Label skip, done;
                LoadState(W9, offsetof(State, loop_count));
                Cbnz(W9, fail_loop);
                g->CBZ(W1, skip);
                g->SUB(W1, W1, 1);
                StoreState(W1, offsetof(State, loop_count));
                StoreStateImm(pc, offsetof(State, loop_start));
                StoreStateImm(pc + imm - 1, offsetof(State, loop_end));
                g->B(done);
                g->l(skip);
                StoreStateImm(pc + imm, offsetof(State, next_pc));
                g->l(done);
                g->MOV(W0, WZR);
                return;
            }
            case 24:
                StoreStateImm(pc + (imm & 0x7fff), offsetof(State, next_pc));
                g->MOV(W0, WZR);
                return;
            case 25: case 26: case 27: case 28: case 29: {
                Label not_taken;
                g->CMP(W1, W2);
                const Cond cond = Condition(op);
                // Taken when the condition equals bit 15.
                g->B((imm & 0x8000) ? Invert(cond) : cond, not_taken);
                const auto offset = std::int32_t(imm << 19) >> 19;
                if (offset == -4096) {
                    StoreStateByte(1, offsetof(State, stopped));
                } else {
                    StoreStateImm(Word(std::int32_t(pc) + offset), offsetof(State, next_pc));
                    StoreStateByte((imm & 0x4000) ? 1 : 0, offsetof(State, cancel_end));
                }
                g->l(not_taken);
                g->MOV(W0, WZR);
                return;
            }
            case 30:
                g->CMP(W1, 0xc00);
                BCond(HS, fail_dread);
                g->LDR(X9, X19, offsetof(State, dram));
                g->LDR(W0, X9, W1, UXTW, 2);
                return;
            case 31:
                g->CMP(W1, 0xc00);
                BCond(HS, fail_dwrite);
                g->LDR(X9, X19, offsetof(State, dram));
                g->STR(W2, X9, W1, UXTW, 2);
                g->MOV(W0, WZR);
                return;
            default:
                g->MOV(W0, WZR);
                return;
            }
        }

        bool far;
        Gen* g = nullptr;
        std::unique_ptr<oaknut::CodeBlock> block;
        std::vector<const void*> table;
        std::vector<Label> labels;
        std::vector<bool> loop_end, after_end;
        Label dispatch, success, leave;
        Label fail_underflow, fail_source, fail_destination, fail_extended, fail_merge, fail_loop,
            fail_dread, fail_dwrite, fail_output, fail_emit, fail_outside, fail_steps;
    };

    std::unique_ptr<Program> program;
    std::uint64_t compiled_generation{~std::uint64_t{0}};
};

} // namespace Tegra
