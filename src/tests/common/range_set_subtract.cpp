// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <random>

#include <catch2/catch_test_macros.hpp>

#include "common/range_sets.h"
#include "common/range_sets.inc"

namespace {

constexpr u64 SPACE = 512;

std::array<bool, SPACE> Contents(const Common::RangeSet<u64>& set) {
    std::array<bool, SPACE> contents{};
    set.ForEach([&](u64 begin, u64 end) {
        for (u64 address = begin; address < end; ++address) {
            contents[address] = true;
        }
    });
    return contents;
}

} // Anonymous namespace

TEST_CASE("RangeSet::Subtract leaves neighbours of the range alone", "[range_sets]") {
    Common::RangeSet<u64> set;
    set.Add(100, 50);
    set.Subtract(150, 10);
    set.Subtract(90, 10);
    set.Subtract(0, 0);
    set.Subtract(120, 0);
    std::array<bool, SPACE> expected{};
    for (u64 address = 100; address < 150; ++address) {
        expected[address] = true;
    }
    REQUIRE(Contents(set) == expected);
    set.Subtract(149, 1);
    set.Subtract(90, 11);
    expected[149] = false;
    expected[100] = false;
    REQUIRE(Contents(set) == expected);

    Common::RangeSet<u64> empty;
    empty.Subtract(0, SPACE);
    REQUIRE(empty.Empty());
}

TEST_CASE("RangeSet::Subtract matches a per-address model", "[range_sets]") {
    std::mt19937_64 rng{0x0400486024DEC000ULL};
    Common::RangeSet<u64> set;
    std::array<bool, SPACE> model{};
    for (int step = 0; step < 20000; ++step) {
        const u64 base = rng() % SPACE;
        const u64 size = rng() % (SPACE - base + 1);
        const bool add = rng() % 3 == 0;
        if (add) {
            set.Add(base, size);
        } else {
            set.Subtract(base, size);
        }
        for (u64 address = base; address < base + size; ++address) {
            model[address] = add;
        }
        REQUIRE(Contents(set) == model);
    }
}
