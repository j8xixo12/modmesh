/*
 * Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <modmesh/view/RStaticMesh.hpp> // Must be the first include.

#include <modmesh/view/common_detail.hpp>

namespace modmesh
{

RStaticMesh::RStaticMesh(std::shared_ptr<StaticMesh> const & static_mesh, Qt3DCore::QNode * parent)
    : Qt3DCore::QEntity(parent)
    , m_geometry(new Qt3DCore::QGeometry(this))
    , m_renderer(new Qt3DRender::QGeometryRenderer())
    , m_material(new Qt3DExtras::QDiffuseSpecularMaterial())
{
    update_geometry(*static_mesh);
    m_renderer->setGeometry(m_geometry);
    m_renderer->setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);
    addComponent(m_renderer);
    addComponent(m_material);
}

void RStaticMesh::update_geometry_impl(StaticMesh const & mh, Qt3DCore::QGeometry * geom)
{
    {
        // Build the Qt node (vertex) coordinate buffer.
        auto * vertices = new Qt3DCore::QAttribute(geom);

        vertices->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
        vertices->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
        vertices->setVertexBaseType(Qt3DCore::QAttribute::Float);
        vertices->setVertexSize(3);
        auto * buf = new Qt3DCore::QBuffer(geom);
        {
            // Copy mesh node coordinates into the Qt buffer.
            QByteArray barray;
            barray.resize(mh.nnode() * 3 * sizeof(float));
            SimpleArray<float> sarr = makeSimpleArray<float>(barray, small_vector<size_t>{mh.nnode(), 3}, /*view*/ true);
            for (uint32_t ind = 0; ind < mh.nnode(); ++ind)
            {
                sarr(ind, 0) = mh.ndcrd(ind, 0);
                sarr(ind, 1) = mh.ndcrd(ind, 1);
                sarr(ind, 2) = (3 == mh.ndim()) ? mh.ndcrd(ind, 2) : 0;
            }
            buf->setData(barray);
        }
        vertices->setBuffer(buf);
        vertices->setByteStride(3 * sizeof(float));
        vertices->setCount(mh.nnode());

        geom->addAttribute(vertices);
    }

    {
        auto * buf = new Qt3DCore::QBuffer(geom);
        {
            QByteArray barray;
            barray.resize(mh.nnode() * 3 * sizeof(float));
            float * ptr = reinterpret_cast<float *>(barray.data());
            for (int i = 0; i < mh.nnode(); ++i)
            {
                ptr[i * 3] = 0.0;
                ptr[i * 3 + 1] = 0.0;
                ptr[i * 3 + 2] = 0.0;
            }
            buf->setData(barray);
        }

        {
            auto * vertices = new Qt3DCore::QAttribute(geom);
            vertices->setName(Qt3DCore::QAttribute::defaultNormalAttributeName());
            vertices->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
            vertices->setVertexBaseType(Qt3DCore::QAttribute::Float);
            vertices->setVertexSize(3);
            vertices->setBuffer(buf);
            vertices->setByteStride(3 * sizeof(float));
            vertices->setCount(mh.nnode());
            geom->addAttribute(vertices);
        }
    }

    {
        // Build the Qt node index buffer.
        auto * indices = new Qt3DCore::QAttribute(geom);

        indices->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
        indices->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);
        auto * buf = new Qt3DCore::QBuffer(geom);
        buf->setData(makeQByteArray(mh.ednds()));
        indices->setBuffer(buf);
        indices->setCount(mh.nedge() * 2);

        geom->addAttribute(indices);
    }
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
