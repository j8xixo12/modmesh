/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/visual/RSurface.hpp> // Must be the first include.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace solvcon
{

RSurface::RSurface(
    SimpleArray<float> const & positions,
    SimpleArray<float> const & normals,
    SimpleArray<uint32_t> const & indices)
{
    if (positions.ndim() != 2 || positions.shape(1) != 3)
    {
        throw std::invalid_argument("RSurface: positions must have shape (nvert, 3)");
    }
    if (normals.ndim() != 2 || normals.shape(0) != positions.shape(0) || normals.shape(1) != 3)
    {
        throw std::invalid_argument("RSurface: normals must match positions");
    }
    if (indices.ndim() != 2 || indices.shape(1) != 3)
    {
        throw std::invalid_argument("RSurface: indices must have shape (ntri, 3)");
    }
    size_t const nvert = static_cast<size_t>(positions.shape(0));
    size_t const ntri = static_cast<size_t>(indices.shape(0));
    size_t const max_bytes = std::numeric_limits<quint32>::max();
    if (nvert > max_bytes / (6 * sizeof(float)) || ntri > max_bytes / (3 * sizeof(uint32_t)))
    {
        throw std::invalid_argument("RSurface: geometry exceeds GPU buffer limits");
    }
    float lo[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    float hi[3] = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
    m_interleaved.reserve(nvert * 6);
    for (size_t i = 0; i < nvert; ++i)
    {
        for (size_t d = 0; d < 3; ++d)
        {
            float const p = positions(i, d);
            if (!std::isfinite(p) || !std::isfinite(normals(i, d)))
            {
                throw std::invalid_argument("RSurface: attributes must be finite");
            }
            m_interleaved.push_back(p);
            lo[d] = std::min(lo[d], p);
            hi[d] = std::max(hi[d], p);
        }
        QVector3D const normal(normals(i, 0), normals(i, 1), normals(i, 2));
        if (normal.isNull())
        {
            throw std::invalid_argument("RSurface: normals must be nonzero");
        }
        for (size_t d = 0; d < 3; ++d)
        {
            m_interleaved.push_back(normals(i, d));
        }
    }
    if (nvert > 0)
    {
        m_lo = QVector3D(lo[0], lo[1], lo[2]);
        m_hi = QVector3D(hi[0], hi[1], hi[2]);
    }
    m_indices.reserve(ntri * 3);
    for (size_t i = 0; i < ntri; ++i)
    {
        for (size_t d = 0; d < 3; ++d)
        {
            uint32_t const index = indices(i, d);
            if (index >= nvert)
            {
                throw std::invalid_argument("RSurface: index out of range");
            }
            m_indices.push_back(index);
        }
    }
}

RSurface::RSurface(
    SimpleArray<float> const & positions,
    SimpleArray<float> const & normals,
    SimpleArray<float> const & texcoords,
    SimpleArray<uint32_t> const & indices,
    QImage const & image,
    int alpha_mode,
    float alpha_cutoff,
    float alpha_factor,
    int wrap_s,
    int wrap_t)
    : RSurface(positions, normals, indices)
{
    if (texcoords.ndim() != 2 || texcoords.shape(0) != positions.shape(0) || texcoords.shape(1) != 2)
    {
        throw std::invalid_argument("RSurface: texcoords must have shape (nvert, 2)");
    }
    if (image.isNull() || image.width() > 8192 || image.height() > 8192)
    {
        throw std::invalid_argument("RSurface: expected an image up to 8192 x 8192");
    }
    if (alpha_mode < 0 || alpha_mode > 2 || !std::isfinite(alpha_cutoff) || alpha_cutoff < 0 || alpha_cutoff > 1 ||
        !std::isfinite(alpha_factor) || alpha_factor < 0 || alpha_factor > 1)
    {
        throw std::invalid_argument("RSurface: invalid alpha mode, cutoff, or factor");
    }
    for (int wrap : {wrap_s, wrap_t})
    {
        if (wrap != 10497 && wrap != 33071 && wrap != 33648)
        {
            throw std::invalid_argument("RSurface: invalid texture wrap mode");
        }
    }
    size_t const nvert = static_cast<size_t>(positions.shape(0));
    if (nvert > std::numeric_limits<quint32>::max() / (8 * sizeof(float)))
    {
        throw std::invalid_argument("RSurface: textured geometry exceeds GPU limits");
    }
    SimpleCollector<float> textured;
    textured.reserve(nvert * 8);
    for (size_t i = 0; i < nvert; ++i)
    {
        for (size_t d = 0; d < 6; ++d)
        {
            textured.push_back(m_interleaved.data()[i * 6 + d]);
        }
        for (size_t d = 0; d < 2; ++d)
        {
            if (!std::isfinite(texcoords(i, d)))
            {
                throw std::invalid_argument("RSurface: texture coordinates must be finite");
            }
            textured.push_back(texcoords(i, d));
        }
    }
    m_interleaved = std::move(textured);
    m_image = image.convertToFormat(QImage::Format_RGBA8888);
    m_alpha_mode = alpha_mode;
    m_alpha_cutoff = alpha_cutoff;
    m_alpha_factor = alpha_factor;
    m_wrap_s = wrap_s;
    m_wrap_t = wrap_t;
}

QRhiVertexInputLayout RSurface::vertexInputLayout() const
{
    QRhiVertexInputLayout layout;
    quint32 const stride = static_cast<quint32>((m_image.isNull() ? 6 : 8) * sizeof(float));
    layout.setBindings({{stride}});
    layout.setAttributes({
        {0, 0, QRhiVertexInputAttribute::Float3, 0},
        {0, 1, QRhiVertexInputAttribute::Float3, 3 * sizeof(float)},
    });
    if (!m_image.isNull())
    {
        layout.setAttributes({
            {0, 0, QRhiVertexInputAttribute::Float3, 0},
            {0, 1, QRhiVertexInputAttribute::Float3, 3 * sizeof(float)},
            {0, 2, QRhiVertexInputAttribute::Float2, 6 * sizeof(float)},
        });
    }
    return layout;
}

void RSurface::createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch)
{
    if (!hasGeometry())
    {
        return;
    }
    quint32 const vbytes = static_cast<quint32>(m_interleaved.size() * sizeof(float));
    m_vbuf.reset(rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::VertexBuffer, vbytes));
    m_vbuf->create();
    batch->updateDynamicBuffer(m_vbuf.get(), 0, vbytes, m_interleaved.data());
    m_geometry_dirty = false;
    m_vertex_count = static_cast<quint32>(m_interleaved.size() / (m_image.isNull() ? 6 : 8));
    quint32 const ibytes = static_cast<quint32>(m_indices.size() * sizeof(uint32_t));
    m_ibuf.reset(rhi->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::IndexBuffer, ibytes));
    m_ibuf->create();
    batch->uploadStaticBuffer(m_ibuf.get(), m_indices.data());
    m_index_count = static_cast<quint32>(m_indices.size());
}

bool RSurface::updateGeometry(RSurface && surface)
{
    if (!hasGeometry() || !m_image.isNull() || !surface.m_image.isNull() ||
        m_interleaved.size() != surface.m_interleaved.size() ||
        m_indices.size() != surface.m_indices.size() ||
        !std::equal(m_indices.data(), m_indices.data() + m_indices.size(), surface.m_indices.data()))
    {
        return false;
    }
    m_interleaved = std::move(surface.m_interleaved);
    m_lo = surface.m_lo;
    m_hi = surface.m_hi;
    m_geometry_dirty = true;
    return true;
}

void RSurface::updateUniform(QRhiResourceUpdateBatch * batch, QMatrix4x4 const & view_proj)
{
    if (m_geometry_dirty && m_vbuf)
    {
        batch->updateDynamicBuffer(m_vbuf.get(), 0, static_cast<quint32>(m_interleaved.size() * sizeof(float)), m_interleaved.data());
        m_geometry_dirty = false;
    }
    RDrawable::updateUniform(batch, view_proj);
}

void RSurface::extendBindings(
    QRhi * rhi, QRhiResourceUpdateBatch * batch, std::vector<QRhiShaderResourceBinding> & bindings)
{
    if (m_image.isNull())
    {
        return;
    }
    auto address = [](int wrap)
    { return wrap == 33071 ? QRhiSampler::ClampToEdge : wrap == 33648 ? QRhiSampler::Mirror
                                                                      : QRhiSampler::Repeat; };
    m_sampler.reset(rhi->newSampler(
        QRhiSampler::Linear, QRhiSampler::Linear, QRhiSampler::Linear, address(m_wrap_s), address(m_wrap_t)));
    m_sampler->create();
    m_texture.reset(rhi->newTexture(QRhiTexture::RGBA8, m_image.size(), 1, QRhiTexture::MipMapped | QRhiTexture::UsedWithGenerateMips));
    m_texture->create();
    batch->uploadTexture(m_texture.get(), m_image);
    batch->generateMips(m_texture.get());
    m_settings.reset(rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, 16));
    m_settings->create();
    float const settings[4] = {m_alpha_cutoff, static_cast<float>(m_alpha_mode), m_alpha_factor, 0};
    batch->updateDynamicBuffer(m_settings.get(), 0, 16, settings);
    bindings.push_back(QRhiShaderResourceBinding::sampledTexture(
        1, QRhiShaderResourceBinding::FragmentStage, m_texture.get(), m_sampler.get()));
    bindings.push_back(QRhiShaderResourceBinding::uniformBuffer(
        2, QRhiShaderResourceBinding::FragmentStage, m_settings.get()));
}

void RSurface::release()
{
    m_texture.reset();
    m_sampler.reset();
    m_settings.reset();
    RDrawable::release();
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
