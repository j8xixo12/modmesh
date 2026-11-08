#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/visual/RDrawable.hpp>

#include <solvcon/solvcon.hpp>

#include <QImage>

namespace solvcon
{

/// An indexed triangle surface with authored per-vertex normals.
class RSurface : public RDrawable
{
public:

    RSurface(
        SimpleArray<float> const & positions,
        SimpleArray<float> const & normals,
        SimpleArray<uint32_t> const & indices);

    RSurface(
        SimpleArray<float> const & positions,
        SimpleArray<float> const & normals,
        SimpleArray<float> const & texcoords,
        SimpleArray<uint32_t> const & indices,
        QImage const & image,
        int alpha_mode,
        float alpha_cutoff,
        float alpha_factor,
        int wrap_s,
        int wrap_t);

    void release() override;
    bool translucent() const override { return m_alpha_mode == 2 || RDrawable::translucent(); }

    bool hasGeometry() const { return m_indices.size() > 0; }
    QVector3D bboxLo() const { return m_lo; }
    QVector3D bboxHi() const { return m_hi; }

protected:

    RMaterial::Kind materialKind() const override { return m_image.isNull() ? RMaterial::Kind::Lit : RMaterial::Kind::LitTextured; }
    QRhiGraphicsPipeline::Topology topology() const override { return QRhiGraphicsPipeline::Triangles; }
    QRhiVertexInputLayout vertexInputLayout() const override;
    void createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch) override;
    void extendBindings(
        QRhi * rhi, QRhiResourceUpdateBatch * batch, std::vector<QRhiShaderResourceBinding> & bindings) override;

private:

    SimpleCollector<float> m_interleaved;
    SimpleCollector<uint32_t> m_indices;
    QVector3D m_lo;
    QVector3D m_hi;
    QImage m_image;
    int m_alpha_mode = 0;
    float m_alpha_cutoff = 0.5f;
    float m_alpha_factor = 1.0f;
    int m_wrap_s = 10497;
    int m_wrap_t = 10497;
    std::unique_ptr<QRhiTexture> m_texture;
    std::unique_ptr<QRhiSampler> m_sampler;
    std::unique_ptr<QRhiBuffer> m_settings;

}; /* end class RSurface */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
