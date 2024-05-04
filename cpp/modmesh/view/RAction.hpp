#pragma once

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
 * A*RE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <modmesh/view/common_detail.hpp> // Must be the first include.

#include <QAction>
#include <QWidget>
#include <QPainter>
#include <QList>
#include <QString>

namespace modmesh
{

class RAction
    : public QAction
{
public:

    RAction(
        QString const & text,
        QString const & tipText,
        std::function<void(void)> callback,
        QObject * parent = nullptr);
}; /* end class RAction */

class RAppAction
    : public QAction
{
public:

    RAppAction(
        QString const & text,
        QString const & tipText,
        QString const & appName,
        QObject * parent = nullptr);

    void run();

    QString const & appName() const { return m_appName; }

private:

    QString m_appName;

}; /* end class RAppAction */

class XYPlot : public QWidget {
public:
    XYPlot(QWidget *parent = nullptr) : QWidget(parent) {}

    // Set axis titles
    void setAxisTitles(const QString &xTitle, const QString &yTitle) {
        xAxisTitle = xTitle;
        yAxisTitle = yTitle;
        update();
    }

    // Set axis tick size
    void setAxisTickSize(int tickSize) {
        axisTickSize = tickSize;
        update();
    }

    // Set line width
    void setLineWidth(int width) {
        lineWidth = width;
        update();
    }

    // Set data
    void setData(const QList<QLineF> &data) {
        dataLines = data;
        update();
    }

    // Set plot title
    void setPlotTitle(const QString &title) {
        plotTitle = title;
        update();
    }

    // Set legend
    void setLegend(const QString &legend) {
        plotLegend = legend;
        update();
    }

    // Replot
    void replot() {
        paintEvent(nullptr);
    }

protected:
    void paintEvent(QPaintEvent *event) override {
        Q_UNUSED(event);

        QPainter painter(this);

        // Draw plot title
        painter.drawText(rect(), Qt::AlignTop | Qt::AlignHCenter, plotTitle);

        // Draw axis titles
        painter.drawText(rect(), Qt::AlignBottom | Qt::AlignHCenter, xAxisTitle);
        painter.drawText(rect(), Qt::AlignLeft | Qt::AlignVCenter, yAxisTitle);

        // Draw axes
        int margin = 40;
        int width = this->width() - 2 * margin;
        int height = this->height() - 2 * margin;
        painter.translate(margin, this->height() - margin);
        painter.scale(1, -1); // Flip y-axis
        painter.drawLine(0, 0, width, 0); // X-axis
        painter.drawLine(0, 0, 0, height); // Y-axis

        // Draw data points
        painter.setPen(Qt::blue);
        painter.setBrush(Qt::blue);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.drawLines(dataLines);
        // for (const QPointF &point : dataLines) {
        //     QPoint pixelPos(point.x() * width, -point.y() * height);
        //     painter.drawEllipse(pixelPos, lineWidth / 2, lineWidth / 2);
        // }

        // Draw legend
        painter.drawText(rect(), Qt::AlignBottom | Qt::AlignRight, plotLegend);
        qDebug() << "test app paintEvent";
    }

private:
    QString xAxisTitle;
    QString yAxisTitle;
    int axisTickSize = 5;
    int lineWidth = 2;
    QList<QLineF> dataLines;
    QString plotTitle;
    QString plotLegend;
};


} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
