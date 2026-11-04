# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(SOLVCON_MULTIBODY_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/multibody.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kinematics.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linear.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Math.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Body.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CamProfile.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DriveFunction.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Joint.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ForceElement.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GearCoupler.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DynamicsSystem.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ModalAnalysis.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_MULTIBODY_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/linear.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Math.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Body.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CamProfile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DriveFunction.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Joint.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ForceElement.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GearCoupler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DynamicsSystem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Integrators.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ModalAnalysis.cpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_MULTIBODY_PYMODHEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/multibody_pymod.hpp
    CACHE FILEPATH "" FORCE)

set(SOLVCON_MULTIBODY_PYMODSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/multibody_pymod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pymod/wrap_multibody.cpp
    CACHE FILEPATH "" FORCE)

# The engine needs SVD, QR, and GEEV from the vendor LAPACK; without it the
# Python submodule is still registered but empty (see multibody_pymod.cpp).
set(SOLVCON_MULTIBODY_FILES
    ${SOLVCON_MULTIBODY_HEADERS}
    ${SOLVCON_MULTIBODY_SOURCES}
    ${SOLVCON_MULTIBODY_PYMODHEADERS}
    ${SOLVCON_MULTIBODY_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
