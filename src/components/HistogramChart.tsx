import { useMemo } from 'react'

// Simple histogram chart component
export function HistogramChart({ data }: { data: any }) {
  const maxValue = useMemo(() => {
    const maxValueRaw = Math.max(
      0,
      ...(data.histogram_r || []),
      ...(data.histogram_g || []),
      ...(data.histogram_b || [])
    )
    // Avoid divide-by-zero (e.g., empty histograms or all-zero arrays)
    return maxValueRaw > 0 ? maxValueRaw : 1
  }, [data])
  const bins = data.bins || 256
  const height = 200
  
  return (
    <div style={{ position: 'relative', height: `${height}px`, width: '100%' }}>
      <svg width="100%" height={height} style={{ border: '1px solid #ddd', borderRadius: '4px', backgroundColor: '#fff' }}>
        {/* Draw histograms */}
        {data.histogram_r && data.histogram_r.map((value: number, i: number) => {
          const barHeight = (value / maxValue) * (height - 30)
          const x = (i / bins) * 100
          const width = (100 / bins)
          return (
            <rect
              key={`r-${i}`}
              x={`${x}%`}
              y={height - 30 - barHeight}
              width={`${width}%`}
              height={barHeight}
              fill="rgba(255, 0, 0, 0.6)"
            />
          )
        })}
        {data.histogram_g && data.histogram_g.map((value: number, i: number) => {
          const barHeight = (value / maxValue) * (height - 30)
          const x = (i / bins) * 100
          const width = (100 / bins)
          return (
            <rect
              key={`g-${i}`}
              x={`${x}%`}
              y={height - 30 - barHeight}
              width={`${width}%`}
              height={barHeight}
              fill="rgba(0, 200, 0, 0.6)"
            />
          )
        })}
        {data.histogram_b && data.histogram_b.map((value: number, i: number) => {
          const barHeight = (value / maxValue) * (height - 30)
          const x = (i / bins) * 100
          const width = (100 / bins)
          return (
            <rect
              key={`b-${i}`}
              x={`${x}%`}
              y={height - 30 - barHeight}
              width={`${width}%`}
              height={barHeight}
              fill="rgba(0, 0, 255, 0.6)"
            />
          )
        })}
        {/* X-axis labels */}
        <text x="5%" y={height - 10} fontSize="11" fill="#333" fontWeight="500">0</text>
        <text x="50%" y={height - 10} fontSize="11" fill="#333" textAnchor="middle" fontWeight="500">128</text>
        <text x="95%" y={height - 10} fontSize="11" fill="#333" textAnchor="end" fontWeight="500">255</text>
        {/* Legend */}
        <g transform={`translate(10, 10)`}>
          <rect x="0" y="0" width="14" height="14" fill="rgba(255, 0, 0, 0.7)" stroke="#333" strokeWidth="0.5" />
          <text x="18" y="11" fontSize="12" fill="#333" fontWeight="500">Red</text>
          <rect x="55" y="0" width="14" height="14" fill="rgba(0, 200, 0, 0.7)" stroke="#333" strokeWidth="0.5" />
          <text x="73" y="11" fontSize="12" fill="#333" fontWeight="500">Green</text>
          <rect x="120" y="0" width="14" height="14" fill="rgba(0, 0, 255, 0.7)" stroke="#333" strokeWidth="0.5" />
          <text x="138" y="11" fontSize="12" fill="#333" fontWeight="500">Blue</text>
        </g>
      </svg>
    </div>
  )
}
