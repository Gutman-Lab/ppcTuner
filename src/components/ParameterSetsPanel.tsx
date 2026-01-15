import { useCallback, useState } from 'react'
import type { SavedParameterSet, PpcResult } from '../types'

interface ParameterSetsPanelProps {
  savedParameterSets: SavedParameterSet[]
  setSavedParameterSets: (sets: SavedParameterSet[]) => void
  nextSetId: number
  setNextSetId: (id: number) => void
  ppcData: PpcResult | null
  itemId: string
  itemName?: string
  currentViewport: { x: number, y: number, width: number, height: number, zoom: number } | null
  hueValue: number
  hueWidth: number
  saturationMinimum: number
  intensityUpperLimit: number
  intensityWeakThreshold: number
  intensityStrongThreshold: number
  intensityLowerLimit: number
  setHueValue: (value: number) => void
  setHueWidth: (value: number) => void
  setSaturationMinimum: (value: number) => void
  setIntensityUpperLimit: (value: number) => void
  setIntensityWeakThreshold: (value: number) => void
  setIntensityStrongThreshold: (value: number) => void
  setIntensityLowerLimit: (value: number) => void
}

export function ParameterSetsPanel({
  savedParameterSets,
  setSavedParameterSets,
  nextSetId,
  setNextSetId,
  ppcData,
  itemId,
  itemName,
  currentViewport,
  hueValue,
  hueWidth,
  saturationMinimum,
  intensityUpperLimit,
  intensityWeakThreshold,
  intensityStrongThreshold,
  intensityLowerLimit,
  setHueValue,
  setHueWidth,
  setSaturationMinimum,
  setIntensityUpperLimit,
  setIntensityWeakThreshold,
  setIntensityStrongThreshold,
  setIntensityLowerLimit,
}: ParameterSetsPanelProps) {
  const [isCollapsed, setIsCollapsed] = useState(true)
  const [isSaving, setIsSaving] = useState(false)

  const generatePreviewDataUrl = useCallback(async (set: SavedParameterSet) => {
    const previewWidth = 320
    const overlayOpacity = 0.7

    const baseThumbUrl = `/api/images/${itemId}/thumbnail?width=${previewWidth}`
    const overlayParams = new URLSearchParams({
      item_id: itemId,
      method: 'hsi',
      thumbnail_width: previewWidth.toString(),
      hue_value: set.hueValue.toString(),
      hue_width: set.hueWidth.toString(),
      saturation_minimum: set.saturationMinimum.toString(),
      intensity_upper_limit: set.intensityUpperLimit.toString(),
      intensity_weak_threshold: set.intensityWeakThreshold.toString(),
      intensity_strong_threshold: set.intensityStrongThreshold.toString(),
      intensity_lower_limit: set.intensityLowerLimit.toString(),
      // Preview is informational: show all classes
      show_weak: 'true',
      show_plain: 'true',
      show_strong: 'true',
      color_scheme: 'blue-green-red',
    })
    const overlayUrl = `/api/ppc/label-image?${overlayParams.toString()}`

    const blobToImage = async (blob: Blob): Promise<HTMLImageElement> => {
      const url = URL.createObjectURL(blob)
      try {
        const img = new Image()
        img.decoding = 'async'
        img.src = url
        await new Promise<void>((resolve, reject) => {
          img.onload = () => resolve()
          img.onerror = () => reject(new Error('Failed to load image'))
        })
        return img
      } finally {
        URL.revokeObjectURL(url)
      }
    }

    const [baseResp, overlayResp] = await Promise.all([fetch(baseThumbUrl), fetch(overlayUrl)])
    if (!baseResp.ok) throw new Error(`Failed to fetch thumbnail: ${baseResp.status}`)
    if (!overlayResp.ok) throw new Error(`Failed to fetch overlay: ${overlayResp.status}`)

    const [baseImg, overlayImg] = await Promise.all([
      blobToImage(await baseResp.blob()),
      blobToImage(await overlayResp.blob()),
    ])

    const canvas = document.createElement('canvas')
    canvas.width = baseImg.naturalWidth || baseImg.width
    canvas.height = baseImg.naturalHeight || baseImg.height
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Canvas not available')

    ctx.drawImage(baseImg, 0, 0, canvas.width, canvas.height)
    ctx.globalAlpha = overlayOpacity
    ctx.drawImage(overlayImg, 0, 0, canvas.width, canvas.height)
    ctx.globalAlpha = 1

    return {
      dataUrl: canvas.toDataURL('image/jpeg', 0.85),
      width: canvas.width,
      height: canvas.height,
      overlayOpacity,
    }
  }, [itemId])
  
  return (
    <div style={{ 
      flex: '0 0 280px',
      border: '1px solid #ddd', 
      borderRadius: '8px', 
      padding: '1rem',
      backgroundColor: '#fff',
      maxHeight: isCollapsed ? 'auto' : '80vh',
      overflowY: isCollapsed ? 'hidden' : 'auto'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: isCollapsed ? 0 : '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flex: 1 }}>
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '0.25rem',
              display: 'flex',
              alignItems: 'center',
              color: '#666',
              fontSize: '0.875rem'
            }}
            title={isCollapsed ? 'Expand' : 'Collapse'}
          >
            {isCollapsed ? '▶' : '▼'}
          </button>
          <h3 style={{ margin: 0, fontSize: '1rem', color: '#333' }}>Parameter Sets</h3>
        </div>
        {!isCollapsed && (
          <button
          onClick={async () => {
            if (!ppcData) {
              alert('Please compute PPC first before saving parameters')
              return
            }
            if (!itemId) {
              alert('No image selected')
              return
            }
            setIsSaving(true)
            const newSet: SavedParameterSet = {
              id: `set-${nextSetId}`,
              name: `Run ${nextSetId}`,
              timestamp: Date.now(),
              method: 'hsi',
              itemId,
              itemName,
              savedFrom: currentViewport ? { viewport: currentViewport } : undefined,
              hueValue,
              hueWidth,
              saturationMinimum,
              intensityUpperLimit,
              intensityWeakThreshold,
              intensityStrongThreshold,
              intensityLowerLimit,
              results: ppcData,
              visible: true,
            }
            setSavedParameterSets([...savedParameterSets, newSet])
            setNextSetId(nextSetId + 1)

            try {
              const preview = await generatePreviewDataUrl(newSet)
              setSavedParameterSets((prev) => prev.map((s) => (
                s.id === newSet.id ? {
                  ...s,
                  preview: {
                    dataUrl: preview.dataUrl,
                    width: preview.width,
                    height: preview.height,
                    overlayOpacity: preview.overlayOpacity,
                    createdAt: Date.now(),
                  }
                } : s
              )))
            } catch (e) {
              console.warn('Failed to generate parameter set preview:', e)
            } finally {
              setIsSaving(false)
            }
          }}
          disabled={!ppcData || isSaving}
          style={{
            padding: '0.4rem 0.8rem',
            fontSize: '0.75rem',
            backgroundColor: (ppcData && !isSaving) ? '#4CAF50' : '#ccc',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: (ppcData && !isSaving) ? 'pointer' : 'not-allowed',
            opacity: (ppcData && !isSaving) ? 1 : 0.6
          }}
        >
          {isSaving ? 'Saving...' : 'Save Current'}
          </button>
        )}
      </div>
      
      {!isCollapsed && (
        <>
          {savedParameterSets.length === 0 ? (
        <p style={{ fontSize: '0.875rem', color: '#666', fontStyle: 'italic', textAlign: 'center', marginTop: '2rem' }}>
          No saved parameter sets.<br />
          Compute PPC and click "Save Current" to add one.
        </p>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {savedParameterSets.map((set) => (
            <div
              key={set.id}
              style={{
                border: '1px solid #ddd',
                borderRadius: '4px',
                padding: '0.75rem',
                backgroundColor: set.visible ? '#f0f8ff' : '#f9f9f9',
                fontSize: '0.875rem'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <input
                  type="text"
                  value={set.name}
                  onChange={(e) => {
                    setSavedParameterSets(savedParameterSets.map(s => 
                      s.id === set.id ? { ...s, name: e.target.value } : s
                    ))
                  }}
                  style={{
                    fontSize: '0.875rem',
                    fontWeight: 'bold',
                    color: '#333',
                    border: '1px solid #ddd',
                    borderRadius: '3px',
                    padding: '0.2rem 0.4rem',
                    flex: 1,
                    marginRight: '0.5rem'
                  }}
                  onBlur={(e) => {
                    if (!e.target.value.trim()) {
                      setSavedParameterSets(savedParameterSets.map(s => 
                        s.id === set.id ? { ...s, name: `Run ${s.id.split('-')[1]}` } : s
                      ))
                    }
                  }}
                />
                <div style={{ display: 'flex', gap: '0.25rem', alignItems: 'center' }}>
                  <button
                    onClick={() => {
                      setHueValue(set.hueValue)
                      setHueWidth(set.hueWidth)
                      setSaturationMinimum(set.saturationMinimum)
                      setIntensityUpperLimit(set.intensityUpperLimit)
                      setIntensityWeakThreshold(set.intensityWeakThreshold)
                      setIntensityStrongThreshold(set.intensityStrongThreshold)
                      setIntensityLowerLimit(set.intensityLowerLimit)
                    }}
                    style={{
                      padding: '0.2rem 0.4rem',
                      fontSize: '0.7rem',
                      backgroundColor: '#2196F3',
                      color: 'white',
                      border: 'none',
                      borderRadius: '3px',
                      cursor: 'pointer'
                    }}
                    title="Load these parameters"
                  >
                    Load
                  </button>
                  <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={set.visible}
                      onChange={(e) => {
                        setSavedParameterSets(savedParameterSets.map(s => 
                          s.id === set.id ? { ...s, visible: e.target.checked } : s
                        ))
                      }}
                      style={{ cursor: 'pointer', marginRight: '0.25rem' }}
                    />
                    <span style={{ fontSize: '0.75rem', color: '#666' }}>Show</span>
                  </label>
                  <button
                    onClick={() => {
                      if (confirm(`Delete "${set.name}"?`)) {
                        setSavedParameterSets(savedParameterSets.filter(s => s.id !== set.id))
                      }
                    }}
                    style={{
                      padding: '0.25rem 0.5rem',
                      fontSize: '0.7rem',
                      backgroundColor: '#f44336',
                      color: 'white',
                      border: 'none',
                      borderRadius: '3px',
                      cursor: 'pointer',
                      marginLeft: '0.25rem',
                      fontWeight: 'bold',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      lineHeight: 1
                    }}
                    title="Delete this parameter set"
                  >
                    Delete
                  </button>
                </div>
              </div>
              
              <div style={{ fontSize: '0.75rem', color: '#666', marginBottom: '0.5rem' }}>
                {new Date(set.timestamp).toLocaleTimeString()}
              </div>
              
              {set.results && (
                <div style={{ fontSize: '0.75rem', color: '#333', lineHeight: '1.6' }}>
                  <div><strong>Weak:</strong> {set.results.weak_positive_pixels?.toLocaleString()} ({set.results.weak_percentage?.toFixed(1)}%)</div>
                  <div><strong>Plain:</strong> {set.results.plain_positive_pixels?.toLocaleString()} ({set.results.plain_percentage?.toFixed(1)}%)</div>
                  <div><strong>Strong:</strong> {set.results.strong_positive_pixels?.toLocaleString()} ({set.results.strong_percentage?.toFixed(1)}%)</div>
                  <div style={{ marginTop: '0.25rem', fontWeight: 'bold' }}>
                    <strong>Total:</strong> {set.results.total_positive_pixels?.toLocaleString()} ({set.results.positive_percentage?.toFixed(1)}%)
                  </div>
                </div>
              )}

              {(set.preview?.dataUrl || set.savedFrom?.viewport || set.itemName || set.itemId) && (
                <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid #eee' }}>
                  {set.preview?.dataUrl && (
                    <img
                      src={set.preview.dataUrl}
                      alt="Saved parameter set preview"
                      style={{
                        width: '100%',
                        height: 'auto',
                        borderRadius: '4px',
                        border: '1px solid #e5e5e5',
                        display: 'block',
                      }}
                    />
                  )}
                  <div style={{ marginTop: '0.5rem', fontSize: '0.7rem', color: '#666', lineHeight: 1.4 }}>
                    {(set.itemName || set.itemId) && (
                      <div>
                        <strong>Slide:</strong> {set.itemName || 'Unknown'}{set.itemId ? ` (${set.itemId})` : ''}
                      </div>
                    )}
                    {set.savedFrom?.viewport && (
                      <div>
                        <strong>Viewport:</strong>{' '}
                        x={set.savedFrom.viewport.x.toFixed(4)}, y={set.savedFrom.viewport.y.toFixed(4)}
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              <div style={{ fontSize: '0.7rem', color: '#999', marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid #eee' }}>
                H: {set.hueValue.toFixed(2)}, W: {set.hueWidth.toFixed(2)}<br />
                Weak≥{set.intensityWeakThreshold.toFixed(2)}, Strong&lt;{set.intensityStrongThreshold.toFixed(2)}
              </div>
            </div>
          ))}
        </div>
      )}
        </>
      )}
    </div>
  )
}
