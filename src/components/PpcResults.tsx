import type { PpcResult, IntensityMap } from '../types'

interface PpcResultsProps {
  ppcData: PpcResult | null
  ppcLoading: boolean
  intensityMapLoading: boolean
  intensityMap: IntensityMap | null
}

export function PpcResults({ ppcData, ppcLoading, intensityMapLoading }: PpcResultsProps) {
  if (ppcLoading || intensityMapLoading) {
    return (
      <p style={{ color: '#666' }}>
        {intensityMapLoading ? 'Fetching intensity map...' : 'Computing PPC...'}
      </p>
    )
  }

  if (!ppcData) {
    return <p style={{ color: '#666' }}>Click "Compute PPC" to analyze positive pixels</p>
  }

  return (
    <div style={{ fontSize: '0.875rem', color: '#333', backgroundColor: '#f5f5f5', padding: '0.75rem', borderRadius: '4px' }}>
      <p style={{ margin: '0 0 0.5rem 0', fontWeight: 'bold', color: '#333' }}>Results (HSI):</p>
      <p style={{ margin: '0.25rem 0', color: '#333' }}>
        <strong>Weak:</strong> {ppcData.weak_positive_pixels?.toLocaleString()} pixels ({ppcData.weak_percentage?.toFixed(2)}%)
      </p>
      <p style={{ margin: '0.25rem 0', color: '#333' }}>
        <strong>Plain:</strong> {ppcData.plain_positive_pixels?.toLocaleString()} pixels ({ppcData.plain_percentage?.toFixed(2)}%)
      </p>
      <p style={{ margin: '0.25rem 0', color: '#333' }}>
        <strong>Strong:</strong> {ppcData.strong_positive_pixels?.toLocaleString()} pixels ({ppcData.strong_percentage?.toFixed(2)}%)
      </p>
      <p style={{ margin: '0.5rem 0 0.25rem 0', fontWeight: 'bold', color: '#333' }}>
        <strong>Total Positive:</strong> {ppcData.total_positive_pixels?.toLocaleString()} pixels ({ppcData.positive_percentage?.toFixed(2)}%)
      </p>
      <p style={{ margin: '0.5rem 0 0.25rem 0', color: '#666', fontSize: '0.75rem' }}>
        Tissue: {ppcData.tissue_pixels?.toLocaleString()} | Background: {ppcData.background_pixels?.toLocaleString()}
      </p>
      {ppcData.metrics && (
        <p style={{ margin: '0.5rem 0 0 0', color: '#666', fontSize: '0.75rem' }}>
          {ppcData.metrics.execution_time_seconds === 0.0 ? (
            <span style={{ color: '#4CAF50', fontWeight: 'bold' }}>⚡ Instant classification (client-side, no API call)</span>
          ) : (
            <>
              Time: {ppcData.metrics.execution_time_seconds}s | CPU: {ppcData.metrics.cpu_percent}% | Memory: {ppcData.metrics.memory_mb?.toFixed(1)}MB
            </>
          )}
        </p>
      )}
    </div>
  )
}
