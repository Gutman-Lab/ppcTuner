import { useState } from 'react'
import { HistogramChart } from './HistogramChart'

interface HistogramSectionProps {
  histogramData: any
  histogramLoading: boolean
  minWidth?: string
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
