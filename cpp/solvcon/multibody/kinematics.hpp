#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Fixed-size value types of rigid-body kinematics in Euler parameters (Haug,
 * Computer-Aided Kinematics and Dynamics of Mechanical Systems, Vol. I, Sec.
 * 9.3).  The types live on the stack: the engine runs hundreds of 3-vector
 * operations per body and step, and a heap array per vector would cost more
 * than the arithmetic.
 *
 * Matrices are row-major.  The sums in the products run over the inner index
 * in increasing order.
 *
 * @ingroup group_numerics
 */

#include <cmath>
#include <cstddef>

namespace solvcon
{

namespace multibody
{

/// Column 3-vector (a point, a direction, a velocity, or an angular velocity).
class Vector3
{

public:

    constexpr Vector3()
        : m_v{0.0, 0.0, 0.0}
    {
    }
    constexpr Vector3(double x, double y, double z)
        : m_v{x, y, z}
    {
    }
    Vector3(Vector3 const &) = default;
    Vector3(Vector3 &&) = default;
    Vector3 & operator=(Vector3 const &) = default;
    Vector3 & operator=(Vector3 &&) = default;
    ~Vector3() = default;

    constexpr double operator[](size_t i) const { return m_v[i]; }
    constexpr double & operator[](size_t i) { return m_v[i]; }
    constexpr size_t size() const { return 3; }
    double const * data() const { return m_v; }

    Vector3 & operator+=(Vector3 const & o)
    {
        m_v[0] += o.m_v[0];
        m_v[1] += o.m_v[1];
        m_v[2] += o.m_v[2];
        return *this;
    }
    Vector3 & operator-=(Vector3 const & o)
    {
        m_v[0] -= o.m_v[0];
        m_v[1] -= o.m_v[1];
        m_v[2] -= o.m_v[2];
        return *this;
    }
    Vector3 & operator*=(double s)
    {
        m_v[0] *= s;
        m_v[1] *= s;
        m_v[2] *= s;
        return *this;
    }
    Vector3 & operator/=(double s)
    {
        m_v[0] /= s;
        m_v[1] /= s;
        m_v[2] /= s;
        return *this;
    }

private:

    double m_v[3]; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)

}; /* end class Vector3 */

/**
 * Column 4-vector in Euler-parameter space: the Euler parameters p = [e0, e1,
 * e2, e3] (e0 is the scalar part), their time derivatives, and the
 * generalized forces conjugate to p.
 */
class Quaternion
{

public:

    constexpr Quaternion()
        : m_v{0.0, 0.0, 0.0, 0.0}
    {
    }
    constexpr Quaternion(double e0, double e1, double e2, double e3)
        : m_v{e0, e1, e2, e3}
    {
    }
    Quaternion(Quaternion const &) = default;
    Quaternion(Quaternion &&) = default;
    Quaternion & operator=(Quaternion const &) = default;
    Quaternion & operator=(Quaternion &&) = default;
    ~Quaternion() = default;

    /// The identity rotation p = [1, 0, 0, 0].
    static constexpr Quaternion identity() { return Quaternion(1.0, 0.0, 0.0, 0.0); }

    constexpr double operator[](size_t i) const { return m_v[i]; }
    constexpr double & operator[](size_t i) { return m_v[i]; }
    constexpr size_t size() const { return 4; }
    double const * data() const { return m_v; }

    /// Vector part e = [e1, e2, e3].
    Vector3 vector_part() const { return Vector3(m_v[1], m_v[2], m_v[3]); }

    Quaternion & operator+=(Quaternion const & o)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            m_v[i] += o.m_v[i];
        }
        return *this;
    }
    Quaternion & operator-=(Quaternion const & o)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            m_v[i] -= o.m_v[i];
        }
        return *this;
    }
    Quaternion & operator*=(double s)
    {
        for (double & v : m_v)
        {
            v *= s;
        }
        return *this;
    }
    Quaternion & operator/=(double s)
    {
        for (double & v : m_v)
        {
            v /= s;
        }
        return *this;
    }

private:

    double m_v[4]; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)

}; /* end class Quaternion */

/// Row-major 3-by-3 matrix (rotation matrices, inertia tensors, cross-product matrices).
class Matrix3
{

public:

    constexpr Matrix3()
        : m_v{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    {
    }
    /// Elements in row-major order.
    constexpr Matrix3(double a00, double a01, double a02, double a10, double a11, double a12, double a20, double a21, double a22)
        : m_v{a00, a01, a02, a10, a11, a12, a20, a21, a22}
    {
    }
    Matrix3(Matrix3 const &) = default;
    Matrix3(Matrix3 &&) = default;
    Matrix3 & operator=(Matrix3 const &) = default;
    Matrix3 & operator=(Matrix3 &&) = default;
    ~Matrix3() = default;

    static constexpr Matrix3 identity() { return Matrix3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0); }
    static constexpr Matrix3 diagonal(Vector3 const & d) { return Matrix3(d[0], 0.0, 0.0, 0.0, d[1], 0.0, 0.0, 0.0, d[2]); }

    constexpr double operator()(size_t i, size_t j) const { return m_v[i * 3 + j]; }
    constexpr double & operator()(size_t i, size_t j) { return m_v[i * 3 + j]; }

    Matrix3 transpose() const
    {
        Matrix3 out;
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                out(j, i) = (*this)(i, j);
            }
        }
        return out;
    }

    double trace() const { return m_v[0] + m_v[4] + m_v[8]; }

    Matrix3 & operator+=(Matrix3 const & o)
    {
        for (size_t k = 0; k < 9; ++k)
        {
            m_v[k] += o.m_v[k];
        }
        return *this;
    }
    Matrix3 & operator-=(Matrix3 const & o)
    {
        for (size_t k = 0; k < 9; ++k)
        {
            m_v[k] -= o.m_v[k];
        }
        return *this;
    }
    Matrix3 & operator*=(double s)
    {
        for (double & v : m_v)
        {
            v *= s;
        }
        return *this;
    }

private:

    double m_v[9]; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)

}; /* end class Matrix3 */

/// Row-major 3-by-4 matrix: the Haug E and G matrices and the derivative d(A s')/dp.
class Matrix34
{

public:

    constexpr Matrix34()
        : m_v{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    {
    }
    Matrix34(Matrix34 const &) = default;
    Matrix34(Matrix34 &&) = default;
    Matrix34 & operator=(Matrix34 const &) = default;
    Matrix34 & operator=(Matrix34 &&) = default;
    ~Matrix34() = default;

    constexpr double operator()(size_t i, size_t j) const { return m_v[i * 4 + j]; }
    constexpr double & operator()(size_t i, size_t j) { return m_v[i * 4 + j]; }

    void set_column(size_t j, Vector3 const & c)
    {
        m_v[j] = c[0];
        m_v[4 + j] = c[1];
        m_v[8 + j] = c[2];
    }

private:

    double m_v[12]; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)

}; /* end class Matrix34 */

/// Row-major 4-by-4 matrix: the rotational mass block 4 G^T J' G of a body (Haug Eq. 11.1.7).
class Matrix4
{

public:

    constexpr Matrix4()
        : m_v{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    {
    }
    Matrix4(Matrix4 const &) = default;
    Matrix4(Matrix4 &&) = default;
    Matrix4 & operator=(Matrix4 const &) = default;
    Matrix4 & operator=(Matrix4 &&) = default;
    ~Matrix4() = default;

    static Matrix4 identity()
    {
        Matrix4 out;
        for (size_t k = 0; k < 4; ++k)
        {
            out(k, k) = 1.0;
        }
        return out;
    }

    constexpr double operator()(size_t i, size_t j) const { return m_v[i * 4 + j]; }
    constexpr double & operator()(size_t i, size_t j) { return m_v[i * 4 + j]; }

private:

    double m_v[16]; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)

}; /* end class Matrix4 */

inline Vector3 operator+(Vector3 a, Vector3 const & b) { return a += b; }
inline Vector3 operator-(Vector3 a, Vector3 const & b) { return a -= b; }
inline Vector3 operator-(Vector3 const & a) { return Vector3(-a[0], -a[1], -a[2]); }
inline Vector3 operator*(double s, Vector3 a) { return a *= s; }
inline Vector3 operator*(Vector3 a, double s) { return a *= s; }
inline Vector3 operator/(Vector3 a, double s) { return a /= s; }

inline double dot(Vector3 const & a, Vector3 const & b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

inline Vector3 cross(Vector3 const & a, Vector3 const & b)
{
    return Vector3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

inline double norm(Vector3 const & a) { return std::sqrt(dot(a, a)); }

/// Unit vector along a; a zero vector is returned unchanged.
inline Vector3 normalized(Vector3 const & a)
{
    double const n = norm(a);
    return (n != 0.0) ? a / n : a;
}

inline Quaternion operator+(Quaternion a, Quaternion const & b) { return a += b; }
inline Quaternion operator-(Quaternion a, Quaternion const & b) { return a -= b; }
inline Quaternion operator-(Quaternion const & a) { return Quaternion(-a[0], -a[1], -a[2], -a[3]); }
inline Quaternion operator*(double s, Quaternion a) { return a *= s; }
inline Quaternion operator*(Quaternion a, double s) { return a *= s; }
inline Quaternion operator/(Quaternion a, double s) { return a /= s; }

inline double dot(Quaternion const & a, Quaternion const & b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; }

inline double norm(Quaternion const & a) { return std::sqrt(dot(a, a)); }

/// Unit 4-vector along a; a zero vector is returned unchanged.
inline Quaternion normalized(Quaternion const & a)
{
    double const n = norm(a);
    return (n != 0.0) ? a / n : a;
}

inline Matrix3 operator+(Matrix3 a, Matrix3 const & b) { return a += b; }
inline Matrix3 operator-(Matrix3 a, Matrix3 const & b) { return a -= b; }
inline Matrix3 operator*(double s, Matrix3 a) { return a *= s; }

inline Vector3 operator*(Matrix3 const & a, Vector3 const & v)
{
    Vector3 out;
    for (size_t i = 0; i < 3; ++i)
    {
        out[i] = a(i, 0) * v[0] + a(i, 1) * v[1] + a(i, 2) * v[2];
    }
    return out;
}

/// a^T v (with a = TBI it maps a body-frame vector to the global frame).
inline Vector3 transpose_mul(Matrix3 const & a, Vector3 const & v)
{
    Vector3 out;
    for (size_t i = 0; i < 3; ++i)
    {
        out[i] = a(0, i) * v[0] + a(1, i) * v[1] + a(2, i) * v[2];
    }
    return out;
}

inline Matrix3 operator*(Matrix3 const & a, Matrix3 const & b)
{
    Matrix3 out;
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            out(i, j) = a(i, 0) * b(0, j) + a(i, 1) * b(1, j) + a(i, 2) * b(2, j);
        }
    }
    return out;
}

/// a^T b.
inline Matrix3 transpose_mul(Matrix3 const & a, Matrix3 const & b)
{
    Matrix3 out;
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            out(i, j) = a(0, i) * b(0, j) + a(1, i) * b(1, j) + a(2, i) * b(2, j);
        }
    }
    return out;
}

/// Cross-product matrix: skew(a) b = a x b.
inline Matrix3 skew(Vector3 const & a) { return Matrix3(0.0, -a[2], a[1], a[2], 0.0, -a[0], -a[1], a[0], 0.0); }

/// Outer product a b^T.
inline Matrix3 outer(Vector3 const & a, Vector3 const & b)
{
    Matrix3 out;
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            out(i, j) = a[i] * b[j];
        }
    }
    return out;
}

inline Vector3 operator*(Matrix34 const & a, Quaternion const & q)
{
    Vector3 out;
    for (size_t i = 0; i < 3; ++i)
    {
        out[i] = a(i, 0) * q[0] + a(i, 1) * q[1] + a(i, 2) * q[2] + a(i, 3) * q[3];
    }
    return out;
}

/// a^T v, equal to the row vector v^T a.
inline Quaternion transpose_mul(Matrix34 const & a, Vector3 const & v)
{
    Quaternion out;
    for (size_t j = 0; j < 4; ++j)
    {
        out[j] = a(0, j) * v[0] + a(1, j) * v[1] + a(2, j) * v[2];
    }
    return out;
}

/// Congruence g^T j g of a 3-by-3 matrix, summed as (g^T j) g.
inline Matrix4 congruence(Matrix34 const & g, Matrix3 const & j)
{
    double gtj[4][3]; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    for (size_t r = 0; r < 4; ++r)
    {
        for (size_t c = 0; c < 3; ++c)
        {
            gtj[r][c] = g(0, r) * j(0, c) + g(1, r) * j(1, c) + g(2, r) * j(2, c);
        }
    }
    Matrix4 out;
    for (size_t r = 0; r < 4; ++r)
    {
        for (size_t c = 0; c < 4; ++c)
        {
            out(r, c) = gtj[r][0] * g(0, c) + gtj[r][1] * g(1, c) + gtj[r][2] * g(2, c);
        }
    }
    return out;
}

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
