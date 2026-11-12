// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Small Mach-O graphs for architecture selection and symbolic loader paths.
#if defined(FIXTURE_DISTINCT_A)
int relay_distinct_a(void) { return 17; }
#elif defined(FIXTURE_DISTINCT_B)
int relay_distinct_b(void) { return 25; }
#elif defined(FIXTURE_DISTINCT_MAIN)
extern int relay_distinct_a(void);
extern int relay_distinct_b(void);
int relay_path_value(void) { return relay_distinct_a() + relay_distinct_b(); }
#elif defined(FIXTURE_LEAF)
int relay_path_leaf(void) { return 42; }
#elif defined(FIXTURE_DECOY)
int relay_path_child(void) { return 99; }
#elif defined(FIXTURE_CHILD)
extern int relay_path_leaf(void);
int relay_path_child(void) { return relay_path_leaf(); }
#else
extern int relay_path_child(void);
int relay_path_value(void) { return relay_path_child(); }
#endif
