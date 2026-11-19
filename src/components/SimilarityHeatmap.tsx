import * as d3 from 'd3';
import { Download } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

interface SimilarityHeatmapProps {
  labels: string[];
  values: number[][];
  algorithmName: string;
  onExportMatrixCsv?: () => void;
}

interface HoverCell {
  rowIdx: number;
  colIdx: number;
  rowLabel: string;
  colLabel: string;
  score: number;
}

export function SimilarityHeatmap({
  labels,
  values,
  algorithmName,
  onExportMatrixCsv,
}: SimilarityHeatmapProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hovered, setHovered] = useState<HoverCell | null>(null);

  useEffect(() => {
    const svgEl = svgRef.current;
    if (!svgEl || labels.length === 0 || values.length === 0) return;

    const n = labels.length;
    const cellSize = n <= 6 ? 62 : n <= 10 ? 46 : n <= 14 ? 36 : 28;
    const marginTop = 88;
    const marginLeft = 116;
    const marginRight = 24;
    const marginBottom = 24;

    const gridSize = n * cellSize;
    const width = marginLeft + gridSize + marginRight;
    const height = marginTop + gridSize + marginBottom;

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${width} ${height}`);

    const truncateLabel = (txt: string, maxLen = 13) =>
      txt.length > maxLen ? `${txt.slice(0, maxLen - 1)}…` : txt;

    const shortLabels = labels.map((l, i) => `${i + 1}. ${truncateLabel(l, 11)}`);

    const xScale = d3
      .scaleBand<number>()
      .domain(d3.range(n))
      .range([0, gridSize])
      .padding(0.06);

    const yScale = d3
      .scaleBand<number>()
      .domain(d3.range(n))
      .range([0, gridSize])
      .padding(0.06);

    const colorScale = d3
      .scaleLinear<string>()
      .domain([0, 40, 75, 100])
      .range(['#071522', '#143F62', '#5BA4D6', '#65C99A'])
      .clamp(true);

    const g = svg
      .append('g')
      .attr('transform', `translate(${marginLeft}, ${marginTop})`);

    // Top axis labels (columns)
    g.append('g')
      .selectAll('text')
      .data(d3.range(n))
      .enter()
      .append('text')
      .attr('x', (i) => (xScale(i) ?? 0) + xScale.bandwidth() / 2)
      .attr('y', -10)
      .attr('text-anchor', 'start')
      .attr(
        'transform',
        (i) =>
          `rotate(-38, ${(xScale(i) ?? 0) + xScale.bandwidth() / 2}, -10)`
      )
      .attr('fill', '#91A4B0')
      .style('font-family', 'JetBrains Mono, monospace')
      .style('font-size', n > 12 ? '10px' : '11px')
      .text((i) => shortLabels[i]);

    // Left axis labels (rows)
    g.append('g')
      .selectAll('text')
      .data(d3.range(n))
      .enter()
      .append('text')
      .attr('x', -10)
      .attr('y', (i) => (yScale(i) ?? 0) + yScale.bandwidth() / 2)
      .attr('dy', '0.35em')
      .attr('text-anchor', 'end')
      .attr('fill', '#91A4B0')
      .style('font-family', 'JetBrains Mono, monospace')
      .style('font-size', n > 12 ? '10px' : '11px')
      .text((i) => shortLabels[i]);

    // Flatten matrix cells for D3 data join
    const cells: Array<{ r: number; c: number; val: number }> = [];
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        cells.push({ r, c, val: values[r]?.[c] ?? 0 });
      }
    }

    const cellGroups = g
      .selectAll('g.heatmap-cell')
      .data(cells)
      .enter()
      .append('g')
      .attr('class', 'heatmap-cell')
      .style('cursor', 'pointer')
      .on('mouseenter', function (_event, d) {
        d3.select(this)
          .select('rect')
          .attr('stroke', '#CFDDD8')
          .attr('stroke-width', 1.5);
        setHovered({
          rowIdx: d.r,
          colIdx: d.c,
          rowLabel: labels[d.r],
          colLabel: labels[d.c],
          score: d.val,
        });
      })
      .on('mouseleave', function () {
        d3.select(this)
          .select('rect')
          .attr('stroke', 'rgba(207,221,216,0.08)')
          .attr('stroke-width', 1);
      });

    cellGroups
      .append('rect')
      .attr('x', (d) => xScale(d.c) ?? 0)
      .attr('y', (d) => yScale(d.r) ?? 0)
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('rx', 4)
      .attr('fill', (d) => colorScale(d.val))
      .attr('stroke', 'rgba(207,221,216,0.08)')
      .attr('stroke-width', 1);

    if (n <= 12) {
      cellGroups
        .append('text')
        .attr('x', (d) => (xScale(d.c) ?? 0) + xScale.bandwidth() / 2)
        .attr('y', (d) => (yScale(d.r) ?? 0) + yScale.bandwidth() / 2)
        .attr('dy', '0.35em')
        .attr('text-anchor', 'middle')
        .attr('pointer-events', 'none')
        .attr('fill', (d) => (d.val >= 58 ? '#010A12' : '#CFDDD8'))
        .style('font-family', 'JetBrains Mono, monospace')
        .style('font-weight', '600')
        .style('font-size', n <= 6 ? '11px' : '10px')
        .text((d) => `${Math.round(d.val)}`);
    }
  }, [labels, values]);

  return (
    <div className="rounded-lg bg-[#071522]/80 border border-[rgba(207,221,216,0.12)] p-5 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h4 className="font-display text-base font-semibold text-[#CFDDD8]">
            Document Similarity Matrix Heatmap
          </h4>
          <p className="text-xs text-[#91A4B0] mt-0.5">
            Pairwise {labels.length}×{labels.length} similarity (%) computed via {algorithmName}.
          </p>
        </div>

        <div className="flex items-center gap-3 self-start sm:self-auto">
          {/* Color Scale Legend */}
          <div className="hidden md:flex items-center gap-2 text-[11px] font-mono text-[#91A4B0]">
            <span>0%</span>
            <div
              aria-hidden="true"
              className="w-24 h-2.5 rounded-sm border border-[rgba(207,221,216,0.15)]"
              style={{
                background:
                  'linear-gradient(90deg, #071522 0%, #143F62 40%, #5BA4D6 75%, #65C99A 100%)',
              }}
            />
            <span>100%</span>
          </div>

          {onExportMatrixCsv && (
            <button
              type="button"
              onClick={onExportMatrixCsv}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#0B1925] hover:bg-[#143F62] border border-[rgba(207,221,216,0.14)] text-xs font-medium text-[#CFDDD8] transition-colors cursor-pointer whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5" />
              Export Matrix CSV
            </button>
          )}
        </div>
      </div>

      {/* D3 SVG Canvas */}
      <div className="overflow-x-auto pb-2">
        <svg
          ref={svgRef}
          className="w-full max-w-[720px] mx-auto h-auto select-none"
          role="img"
          aria-label="D3.js similarity matrix heatmap"
        />
      </div>

      {/* Hovered Cell Readout */}
      <div className="pt-3 border-t border-[rgba(207,221,216,0.08)] flex flex-wrap items-center justify-between gap-2 text-xs min-h-[28px]">
        {hovered ? (
          <>
            <span className="text-[#91A4B0]">
              <strong className="text-[#CFDDD8]">{hovered.rowLabel}</strong>
              <span className="mx-2 text-[#5BA4D6]">↔</span>
              <strong className="text-[#CFDDD8]">{hovered.colLabel}</strong>
            </span>
            <span className="font-mono font-semibold text-[#65C99A] tabular-nums">
              {hovered.score.toFixed(1)}% similarity
            </span>
          </>
        ) : (
          <span className="text-[#647987]">
            Hover any cell in the matrix to inspect exact pair similarity.
          </span>
        )}
      </div>
    </div>
  );
}
