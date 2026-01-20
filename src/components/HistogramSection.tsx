import { useState } from 'react'

interface HistogramSectionProps {
  histogramData: any
  histogramLoading: boolean
  minWidth?: string
}

function HistogramChart({ data }: { data: any }) {
  const maxValueRaw = Math.max(
    0,
    ...(data.histogram_r || []),
    ...(data.histogram_g || []),
    ...(data.histogram_b || [])
  )
  // Avoid divide-by-zero (e.g., empty histograms or all-zero arrays)
  const maxValue = maxValueRaw > 0 ? maxValueRaw : 1
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

export function HistogramSection({
  histogramData,
  histogramLoading,
  minWidth = '300px',
}: HistogramSectionProps) {
  const [isCollapsed, setIsCollapsed] = useState(true) // Collapsed by default

  return (
    <div style={{ 
      border: '1px solid #ddd', 
      borderRadius: '8px', 
      padding: '1rem',
      backgroundColor: '#fff',
      minWidth
    }}>
      <div 
        style={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          cursor: 'pointer',
          userSelect: 'none',
          marginBottom: isCollapsed ? 0 : '1rem'
        }}
        onClick={() => setIsCollapsed(!isCollapsed)}
        title={isCollapsed ? 'Click to expand histogram' : 'Click to collapse histogram'}
      >
        <h3 style={{ margin: 0, color: '#333' }}>Color Histogram</h3>
        <span style={{ 
          fontSize: '0.875rem', 
          color: '#666',
          transition: 'transform 0.2s',
          transform: isCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)',
          display: 'inline-block'
        }}>
          ▼
        </span>
      </div>
      {!isCollapsed && (
        <div>
          {histogramLoading ? (
            <p style={{ color: '#666' }}>Loading histogram...</p>
          ) : histogramData ? (
            <div>
              <HistogramChart data={histogramData} />
              {histogramData.statistics?.mean && histogramData.statistics?.std && (
                <div style={{ marginTop: '1rem', fontSize: '0.875rem', color: '#333', backgroundColor: '#f5f5f5', padding: '0.75rem', borderRadius: '4px' }}>
                  <p style={{ margin: '0 0 0.5rem 0', fontWeight: 'bold', color: '#333' }}>Statistics:</p>
                  <p style={{ margin: '0.25rem 0', color: '#333' }}>
                    <strong>Mean</strong> - R: {Number(histogramData.statistics.mean.r).toFixed(1)}, G: {Number(histogramData.statistics.mean.g).toFixed(1)}, B: {Number(histogramData.statistics.mean.b).toFixed(1)}
                  </p>
                  <p style={{ margin: '0.25rem 0', color: '#333' }}>
                    <strong>Std</strong> - R: {Number(histogramData.statistics.std.r).toFixed(1)}, G: {Number(histogramData.statistics.std.g).toFixed(1)}, B: {Number(histogramData.statistics.std.b).toFixed(1)}
                  </p>
                </div>
              )}
            </div>
          ) : (
            <p style={{ color: '#666' }}>Click an image to load histogram</p>
          )}
        </div>
      )}
    </div>
  )
}
