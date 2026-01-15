import type { IntensityMap } from '../types'

interface AutoComputeControlsProps {
  autoCompute: boolean
  setAutoCompute: (value: boolean) => void
  intensityMap: IntensityMap | null
}

export function AutoComputeControls({ autoCompute, setAutoCompute, intensityMap }: AutoComputeControlsProps) {
  return (
    <>
      {/* Auto-compute toggle */}
      <div style={{ marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem', backgroundColor: '#f0f0f0', borderRadius: '4px' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.875rem', color: '#333' }}>
          <input
            type="checkbox"
            checked={autoCompute}
            onChange={(e) => setAutoCompute(e.target.checked)}
            style={{ cursor: 'pointer' }}
          />
          <span><strong>Auto-compute</strong> (updates in real-time with 500ms debounce)</span>
        </label>
      </div>
      {autoCompute && (
        <div style={{ fontSize: '0.75rem', color: '#666', marginBottom: '0.75rem', padding: '0.5rem', backgroundColor: '#fff3cd', borderRadius: '4px', border: '1px solid #ffc107' }}>
          <strong>Note:</strong> Adjusting HSI parameters will automatically trigger updates (500ms debounce).<br />
          <strong>Full recomputation:</strong> Hue, saturation, intensity bounds (upper/lower limits) - fetches intensity map from backend<br />
          <strong>⚡ Instant adjustment:</strong> Intensity thresholds (weak/strong) - reclassifies in real-time, no API call needed!
          {intensityMap && (
            <span style={{ color: '#4CAF50', fontWeight: 'bold', marginLeft: '0.5rem' }}>
              ✓ Using intensity map ({intensityMap.positive_intensities.length.toLocaleString()} positive pixels)
            </span>
          )}
        </div>
      )}
    </>
  )
}
