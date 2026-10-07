#pragma once

// Shared assert-like helper for per-component unit tests (tests/unit/*Test.cpp).
// Deliberately not a new test-framework dependency (gtest isn't installed on
// this box, and every other test in this repo is already plain assert()-based
// plain g++/mpic++, no external test deps) -- this is just a tiny upgrade over
// bare assert(): a failed CHECK prints what failed and keeps going instead of
// aborting the whole binary, so one run reports every failure, not just the
// first.

#include <cstdlib>
#include <iostream>
#include <string>

namespace tq::test {

inline int& checks_run() {
    static int n = 0;
    return n;
}
inline int& checks_failed() {
    static int n = 0;
    return n;
}

inline void record_check(bool cond, const char* expr, const char* file, int line) {
    ++checks_run();
    if (!cond) {
        ++checks_failed();
        std::cerr << "FAIL [" << file << ":" << line << "] " << expr << "\n";
    }
}

// Returns a process exit code: 0 if every CHECK passed, 1 otherwise. Call
// once at the end of main().
inline int summary(const char* suite_name) {
    int run = checks_run();
    int failed = checks_failed();
    if (failed == 0) {
        std::cout << suite_name << ": PASS (" << run << " checks)\n";
        return 0;
    }
    std::cout << suite_name << ": FAIL (" << failed << "/" << run << " checks failed)\n";
    return 1;
}

} 

#define CHECK(cond) ::tq::test::record_check(static_cast<bool>(cond), #cond, __FILE__, __LINE__)

// Runs `expr`, records a pass iff it throws exactly `exc_type` (or a type
// convertible to it). Records a failure if it throws nothing, or throws
// something else.
#define CHECK_THROWS(expr, exc_type)                                                                                 \
    do {                                                                                                             \
        bool threw_expected = false;                                                                                 \
        try {                                                                                                        \
            (void)(expr);                                                                                            \
        } catch (const exc_type&) {                                                                                  \
            threw_expected = true;                                                                                   \
        } catch (...) {                                                                                              \
            threw_expected = false;                                                                                  \
        }                                                                                                            \
        ::tq::test::record_check(threw_expected, "expected `" #expr "` to throw " #exc_type, __FILE__, __LINE__);    \
    } while (0)

#define CHECK_NOTHROW(expr)                                                                                          \
    do {                                                                                                             \
        bool threw = false;                                                                                          \
        try {                                                                                                        \
            (void)(expr);                                                                                            \
        } catch (...) {                                                                                              \
            threw = true;                                                                                            \
        }                                                                                                            \
        ::tq::test::record_check(!threw, "expected `" #expr "` not to throw", __FILE__, __LINE__);                   \
    } while (0)
