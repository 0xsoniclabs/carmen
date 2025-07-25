// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

#pragma once

// The following has been taken from
//  https://github.com/swansontec/map-macro/blob/master/map.h
// for which all right have been waived.

/*
 * Created by William Swanson in 2012.
 *
 * I, William Swanson, dedicate this work to the public domain.
 * I waive all rights to the work worldwide under copyright law,
 * including all related and neighboring rights,
 * to the extent allowed by law.
 *
 * You can copy, modify, distribute and perform the work,
 * even for commercial purposes, all without asking permission.
 */

#ifndef MAP_H_INCLUDED
#define MAP_H_INCLUDED

#define EVAL0(...) __VA_ARGS__
#define EVAL1(...) EVAL0(EVAL0(EVAL0(__VA_ARGS__)))
#define EVAL2(...) EVAL1(EVAL1(EVAL1(__VA_ARGS__)))
#define EVAL3(...) EVAL2(EVAL2(EVAL2(__VA_ARGS__)))
#define EVAL4(...) EVAL3(EVAL3(EVAL3(__VA_ARGS__)))
#define EVAL(...) EVAL4(EVAL4(EVAL4(__VA_ARGS__)))

#define MAP_END(...)
#define MAP_OUT
#define MAP_COMMA ,

#define MAP_GET_END2() 0, MAP_END
#define MAP_GET_END1(...) MAP_GET_END2
#define MAP_GET_END(...) MAP_GET_END1
#define MAP_NEXT0(test, next, ...) next MAP_OUT
#define MAP_NEXT1(test, next) MAP_NEXT0(test, next, 0)
#define MAP_NEXT(test, next) MAP_NEXT1(MAP_GET_END test, next)

#define MAP0(f, x, peek, ...) f(x) MAP_NEXT(peek, MAP1)(f, peek, __VA_ARGS__)
#define MAP1(f, x, peek, ...) f(x) MAP_NEXT(peek, MAP0)(f, peek, __VA_ARGS__)

#define MAP_LIST_NEXT1(test, next) MAP_NEXT0(test, MAP_COMMA next, 0)
#define MAP_LIST_NEXT(test, next) MAP_LIST_NEXT1(MAP_GET_END test, next)

#define MAP_LIST0(f, x, peek, ...) \
  f(x) MAP_LIST_NEXT(peek, MAP_LIST1)(f, peek, __VA_ARGS__)
#define MAP_LIST1(f, x, peek, ...) \
  f(x) MAP_LIST_NEXT(peek, MAP_LIST0)(f, peek, __VA_ARGS__)

/**
 * Applies the function macro `f` to each of the remaining parameters.
 */
#define MAP(f, ...) EVAL(MAP1(f, __VA_ARGS__, ()()(), ()()(), ()()(), 0))

/**
 * Applies the function macro `f` to each of the remaining parameters and
 * inserts commas between the results.
 */
#define MAP_LIST(f, ...) \
  EVAL(MAP_LIST1(f, __VA_ARGS__, ()()(), ()()(), ()()(), 0))

#endif

// ----------------------------------------------------------------------------

// The following is not related to the content above, but provides additional
// common macro utilities.

// The following set of macros will remove optional paranthesis from an
// expression, by following one of the two expansion paths:
//
//  Without parenthesis:
//    REMOVE_OPTIONAL_PARENTHESIS(expr)
//      -> INTERNAL_REMOVE_PARENT(PEAR expr)
//      -> INTERNAL_REMOVE_PARENT_(PEAR expr)
//      -> INTERNAL_DISAPPEAR expr
//      -> expr
//
//  With parenthesis:
//    REMOVE_OPTIONAL_PARENTHESIS((expr))
//      -> INTERNAL_REMOVE_PARENT(PEAR(expr))
//      -> INTERNAL_REMOVE_PARENT_(PEAR expr)
//      -> INTERNAL_DISAPPEAR expr
//      -> expr

#define REMOVE_OPTIONAL_PARENTHESIS(expr) INTERNAL_REMOVE_PARENT(PEAR expr)
#define PEAR(...) PEAR __VA_ARGS__
#define INTERNAL_REMOVE_PARENT(...) INTERNAL_REMOVE_PARENT_(__VA_ARGS__)
#define INTERNAL_REMOVE_PARENT_(...) INTERNAL_DISAP##__VA_ARGS__
#define INTERNAL_DISAPPEAR
