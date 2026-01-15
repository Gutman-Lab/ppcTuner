import type { PpcResult } from '../types'

interface RegionPpcResultsProps {
  regionPpcData: PpcResult
}

export function RegionPpcResults({ regionPpcData }: RegionPpcResultsProps) {
  return (
    <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#fff', border: '1px solid #ddd', borderRadius: '4px' }}>
      <h4 style={{ marginTop: 0, marginBottom: '0.75rem', color: '#333', fontSize: '1rem', fontWeight: '600' }}>Region PPC Results</h4>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem', fontSize: '0.875rem' }}>
        <div style={{ color: '#333' }}>
          <strong style={{ color: '#222', fontWeight: '600' }}>Total Pixels:</strong> 
          <span style={{ marginLeft: '0.5rem', color: '#555' }}>{regionPpcData.total_pixels.toLocaleString()}</span>
        </div>
        <div style={{ color: '#333' }}>
          <strong style={{ color: '#222', fontWeight: '600' }}>Tissue Pixels:</strong> 
          <span style={{ marginLeft: '0.5rem', color: '#555' }}>{regionPpcData.tissue_pixels.toLocaleString()}</span>
        </div>
        <div style={{ color: '#333' }}>
          <strong style={{ color: '#222', fontWeight: '600' }}>Weak Positive:</strong> 
          <span style={{ marginLeft: '0.5rem', color: '#555' }}>
            {(regionPpcData.weak_positive_pixels ?? 0).toLocaleString()} 
            <span style={{ color: '#666', marginLeft: '0.25rem' }}>({regionPpcData.weak_percentage ?? 0}%)</span>
          </span>
        </div>
        <div style={{ color: '#333' }}>
          <strong style={{ color: '#222', fontWeight: '600' }}>Plain Positive:</strong> 
          <span style={{ marginLeft: '0.5rem', color: '#555' }}>
            {(regionPpcData.plain_positive_pixels ?? 0).toLocaleString()} 
            <span style={{ color: '#666', marginLeft: '0.25rem' }}>({regionPpcData.plain_percentage ?? 0}%)</span>
          </span>
        </div>
        <div style={{ color: '#333' }}>
          <strong style={{ color: '#222', fontWeight: '600' }}>Strong Positive:</strong> 
          <span style={{ marginLeft: '0.5rem', color: '#555' }}>
            {(regionPpcData.strong_positive_pixels ?? 0).toLocaleString()} 
            <span style={{ color: '#666', marginLeft: '0.25rem' }}>({regionPpcData.strong_percentage ?? 0}%)</span>
          </span>
        </div>
        <div style={{ color: '#333' }}>
          <strong style={{ color: '#222', fontWeight: '600' }}>Total Positive:</strong> 
          <span style={{ marginLeft: '0.5rem', color: '#555', fontWeight: '500' }}>
            {(regionPpcData.total_positive_pixels ?? 0).toLocaleString()} 
            <span style={{ color: '#666', marginLeft: '0.25rem' }}>({regionPpcData.positive_percentage ?? 0}%)</span>
          </span>
        </div>
      </div>
    </div>
  )
}
