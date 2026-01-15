interface PpcControlsProps {
  showPpcLabel: boolean
  ppcLabelOpacity: number
  setPpcLabelOpacity: (value: number) => void
  showWeak: boolean
  showPlain: boolean
  showStrong: boolean
  setShowWeak: (value: boolean) => void
  setShowPlain: (value: boolean) => void
  setShowStrong: (value: boolean) => void
  labelColorScheme: string
  setLabelColorScheme: (value: string) => void
}

export function PpcControls({
  showPpcLabel,
  ppcLabelOpacity,
  setPpcLabelOpacity,
  showWeak,
  showPlain,
  showStrong,
  setShowWeak,
  setShowPlain,
  setShowStrong,
  labelColorScheme,
  setLabelColorScheme,
}: PpcControlsProps) {
  if (!showPpcLabel) return null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.875rem', alignItems: 'center', width: '100%' }}>
      {/* Color Legend */}
      <div style={{ 
        display: 'flex', 
        gap: '1rem', 
        alignItems: 'center', 
        padding: '0.5rem',
        backgroundColor: '#f9f9f9',
        borderRadius: '4px',
        flexWrap: 'wrap',
        justifyContent: 'center'
      }}>
        <span style={{ fontWeight: 'bold', color: '#333', marginRight: '0.5rem' }}>Legend:</span>
        {labelColorScheme === 'yellow-orange-red' ? (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <div style={{ width: '16px', height: '16px', backgroundColor: '#FFFF00', border: '1px solid #ccc', borderRadius: '2px' }} />
              <span style={{ color: '#333' }}>Weak</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <div style={{ width: '16px', height: '16px', backgroundColor: '#FFA500', border: '1px solid #ccc', borderRadius: '2px' }} />
              <span style={{ color: '#333' }}>Plain</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <div style={{ width: '16px', height: '16px', backgroundColor: '#FF0000', border: '1px solid #ccc', borderRadius: '2px' }} />
              <span style={{ color: '#333' }}>Strong</span>
            </div>
          </>
        ) : (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <div style={{ width: '16px', height: '16px', backgroundColor: '#0000FF', border: '1px solid #ccc', borderRadius: '2px' }} />
              <span style={{ color: '#333' }}>Weak</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <div style={{ width: '16px', height: '16px', backgroundColor: '#00FF00', border: '1px solid #ccc', borderRadius: '2px' }} />
              <span style={{ color: '#333' }}>Plain</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <div style={{ width: '16px', height: '16px', backgroundColor: '#FF0000', border: '1px solid #ccc', borderRadius: '2px' }} />
              <span style={{ color: '#333' }}>Strong</span>
            </div>
          </>
        )}
      </div>
      
      {/* Opacity Control */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <label style={{ color: '#333' }}>Opacity:</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={ppcLabelOpacity}
          onChange={(e) => setPpcLabelOpacity(parseFloat(e.target.value))}
          style={{ width: '150px' }}
        />
        <span style={{ color: '#666', minWidth: '40px' }}>{(ppcLabelOpacity * 100).toFixed(0)}%</span>
      </div>
      
      {/* Toggle Controls */}
      <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
        <label style={{ color: '#333', display: 'flex', alignItems: 'center', gap: '0.25rem', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showWeak}
            onChange={(e) => setShowWeak(e.target.checked)}
            style={{ cursor: 'pointer' }}
          />
          <span>Weak</span>
        </label>
        <label style={{ color: '#333', display: 'flex', alignItems: 'center', gap: '0.25rem', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showPlain}
            onChange={(e) => setShowPlain(e.target.checked)}
            style={{ cursor: 'pointer' }}
          />
          <span>Plain</span>
        </label>
        <label style={{ color: '#333', display: 'flex', alignItems: 'center', gap: '0.25rem', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showStrong}
            onChange={(e) => setShowStrong(e.target.checked)}
            style={{ cursor: 'pointer' }}
          />
          <span>Strong</span>
        </label>
        <select
          value={labelColorScheme}
          onChange={(e) => setLabelColorScheme(e.target.value)}
          style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem', borderRadius: '4px', border: '1px solid #ddd', marginLeft: '0.5rem' }}
        >
          <option value="blue-green-red">Blue-Green-Red</option>
          <option value="yellow-orange-red">Yellow-Orange-Red</option>
        </select>
      </div>
    </div>
  )
}
