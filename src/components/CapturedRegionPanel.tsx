import { useCallback } from 'react'
import type { Dispatch, SetStateAction } from 'react'
import { PpcLabelOverlayRegion } from './PpcLabelOverlayRegion'
import { RegionPpcResults } from './RegionPpcResults'
import { safeJsonParse } from '../utils/api'
import type { CapturedRegion } from '../types'

interface CapturedRegionPanelProps {
  capturedRegions: CapturedRegion[]
  setCapturedRegions: Dispatch<SetStateAction<CapturedRegion[]>>
  // Current HSI parameters (used for "Recompute")
  hueValue: number
  hueWidth: number
  saturationMinimum: number
  intensityUpperLimit: number
  intensityWeakThreshold: number
  intensityStrongThreshold: number
  intensityLowerLimit: number
  // Current display options (used for "Recompute")
  showWeak: boolean
  showPlain: boolean
  showStrong: boolean
  labelColorScheme: string
}

export function CapturedRegionPanel({
  capturedRegions,
  setCapturedRegions,
  hueValue,
  hueWidth,
  saturationMinimum,
  intensityUpperLimit,
  intensityWeakThreshold,
  intensityStrongThreshold,
  intensityLowerLimit,
  showWeak,
  showPlain,
  showStrong,
  labelColorScheme,
}: CapturedRegionPanelProps) {
  const revokeIfBlobUrl = (url: string | null) => {
    if (url && url.startsWith('blob:')) {
      try { URL.revokeObjectURL(url) } catch { /* noop */ }
    }
  }

  const removeRegion = useCallback((id: string) => {
    setCapturedRegions((prev) => {
      const toRemove = prev.find((r) => r.id === id)
      if (toRemove) revokeIfBlobUrl(toRemove.imageUrl)
      return prev.filter((r) => r.id !== id)
    })
  }, [setCapturedRegions])

  const clearAll = useCallback(() => {
    setCapturedRegions((prev) => {
      prev.forEach((r) => revokeIfBlobUrl(r.imageUrl))
      return []
    })
  }, [setCapturedRegions])

  const recomputeRegion = useCallback(async (r: CapturedRegion) => {
    setCapturedRegions((prev) => prev.map((x) => (
      x.id === r.id ? {
        ...x,
        ppcLoading: true,
        // Recompute uses current global params + display options
        paramsUsed: {
          hueValue,
          hueWidth,
          saturationMinimum,
          intensityUpperLimit,
          intensityWeakThreshold,
          intensityStrongThreshold,
          intensityLowerLimit,
        },
        displayUsed: {
          showWeak,
          showPlain,
          showStrong,
          labelColorScheme,
        },
        // Clear cached overlay because parameters changed
        overlayDataUrl: null,
        showOverlayOnMain: false,
      } : x
    )))

    try {
      const params = new URLSearchParams({
        item_id: r.itemId,
        x: r.region.x.toString(),
        y: r.region.y.toString(),
        width: r.region.width.toString(),
        height: r.region.height.toString(),
        output_width: '1024',
        hue_value: hueValue.toString(),
        hue_width: hueWidth.toString(),
        saturation_minimum: saturationMinimum.toString(),
        intensity_upper_limit: intensityUpperLimit.toString(),
        intensity_weak_threshold: intensityWeakThreshold.toString(),
        intensity_strong_threshold: intensityStrongThreshold.toString(),
        intensity_lower_limit: intensityLowerLimit.toString(),
      })
      const response = await fetch(`/api/ppc/compute-region?${params.toString()}`)
      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Failed to compute PPC on region: ${response.status} ${errorText}`)
      }
      const data = await safeJsonParse(response)
      setCapturedRegions((prev) => prev.map((x) => (
        x.id === r.id ? { ...x, ppcData: data, ppcLoading: false } : x
      )))
    } catch (e) {
      console.error(e)
      setCapturedRegions((prev) => prev.map((x) => (
        x.id === r.id ? { ...x, ppcLoading: false } : x
      )))
      alert(String(e))
    }
  }, [
    setCapturedRegions,
    hueValue,
    hueWidth,
    saturationMinimum,
    intensityUpperLimit,
    intensityWeakThreshold,
    intensityStrongThreshold,
    intensityLowerLimit,
    showWeak,
    showPlain,
    showStrong,
    labelColorScheme,
  ])

  const ensureMainOverlayDataUrl = useCallback(async (r: CapturedRegion) => {
    const params = new URLSearchParams({
      item_id: r.itemId,
      x: r.region.x.toString(),
      y: r.region.y.toString(),
      width: r.region.width.toString(),
      height: r.region.height.toString(),
      output_width: '2048',
      method: 'hsi',
      hue_value: r.paramsUsed.hueValue.toString(),
      hue_width: r.paramsUsed.hueWidth.toString(),
      saturation_minimum: r.paramsUsed.saturationMinimum.toString(),
      intensity_upper_limit: r.paramsUsed.intensityUpperLimit.toString(),
      intensity_weak_threshold: r.paramsUsed.intensityWeakThreshold.toString(),
      intensity_strong_threshold: r.paramsUsed.intensityStrongThreshold.toString(),
      intensity_lower_limit: r.paramsUsed.intensityLowerLimit.toString(),
      show_weak: r.displayUsed.showWeak.toString(),
      show_plain: r.displayUsed.showPlain.toString(),
      show_strong: r.displayUsed.showStrong.toString(),
      color_scheme: r.displayUsed.labelColorScheme,
    })

    const url = `/api/ppc/label-image-region?${params.toString()}`
    const response = await fetch(url)
    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`Failed to load region overlay: ${response.status} ${errorText}`)
    }
    const blob = await response.blob()
    const dataUrl = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader()
      reader.onloadend = () => resolve(reader.result as string)
      reader.onerror = () => reject(new Error('Failed to convert overlay to data URL'))
      reader.readAsDataURL(blob)
    })

    setCapturedRegions((prev) => prev.map((x) => (
      x.id === r.id ? { ...x, overlayDataUrl: dataUrl } : x
    )))
  }, [setCapturedRegions])

  if (capturedRegions.length === 0) return null

  return (
    <div style={{
      flex: '0 0 400px',
      border: '1px solid #ddd',
      borderRadius: '8px',
      padding: '1rem',
      backgroundColor: '#fff',
      maxHeight: '80vh',
      overflowY: 'auto',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h3 style={{ margin: 0, fontSize: '1rem', color: '#333' }}>
          Captured Regions ({capturedRegions.length})
        </h3>
        <button
          onClick={() => {
            if (confirm('Clear all captured regions?')) clearAll()
          }}
          style={{
            padding: '0.25rem 0.5rem',
            fontSize: '0.75rem',
            backgroundColor: '#f44336',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          Clear All
        </button>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
        {capturedRegions.map((r, idx) => (
          <div
            key={r.id}
            style={{
              border: '1px solid #ddd',
              borderRadius: '6px',
              padding: '0.75rem',
              backgroundColor: '#fafafa',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ fontWeight: 700, color: '#333' }}>Region #{idx + 1}</div>
              <button
                onClick={() => removeRegion(r.id)}
                style={{
                  padding: '0.25rem 0.5rem',
                  fontSize: '0.7rem',
                  backgroundColor: '#f44336',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                }}
                title="Remove this captured region"
              >
                Remove
              </button>
            </div>

            <div style={{ marginTop: '0.35rem', fontSize: '0.7rem', color: '#666' }}>
              x={r.region.x.toFixed(4)}, y={r.region.y.toFixed(4)}, w={r.region.width.toFixed(4)}, h={r.region.height.toFixed(4)}
            </div>

            <div style={{ marginTop: '0.6rem', position: 'relative', width: '100%' }}>
              {r.imageUrl ? (
                <img
                  src={r.imageUrl}
                  alt={`Captured region ${idx + 1}`}
                  style={{
                    width: '100%',
                    height: 'auto',
                    borderRadius: '4px',
                    border: '1px solid #e5e5e5',
                    display: 'block',
                  }}
                />
              ) : (
                <div style={{
                  width: '100%',
                  height: '140px',
                  borderRadius: '4px',
                  border: '1px solid #e5e5e5',
                  backgroundColor: '#fff',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#777',
                  fontSize: '0.8rem',
                }}>
                  {r.isCapturing ? 'Capturing...' : 'No image'}
                </div>
              )}

              {r.showOverlayOnThumbnail && r.ppcData && (
                <PpcLabelOverlayRegion
                  itemId={r.itemId}
                  x={r.region.x}
                  y={r.region.y}
                  width={r.region.width}
                  height={r.region.height}
                  method="hsi"
                  opacity={r.overlayOpacity}
                  showWeak={r.displayUsed.showWeak}
                  showPlain={r.displayUsed.showPlain}
                  showStrong={r.displayUsed.showStrong}
                  colorScheme={r.displayUsed.labelColorScheme}
                  hueValue={r.paramsUsed.hueValue}
                  hueWidth={r.paramsUsed.hueWidth}
                  saturationMinimum={r.paramsUsed.saturationMinimum}
                  intensityUpperLimit={r.paramsUsed.intensityUpperLimit}
                  intensityWeakThreshold={r.paramsUsed.intensityWeakThreshold}
                  intensityStrongThreshold={r.paramsUsed.intensityStrongThreshold}
                  intensityLowerLimit={r.paramsUsed.intensityLowerLimit}
                />
              )}
            </div>

            <div style={{ marginTop: '0.6rem', display: 'flex', gap: '0.5rem' }}>
              <button
                onClick={() => void recomputeRegion(r)}
                disabled={r.ppcLoading}
                style={{
                  flex: 1,
                  padding: '0.45rem 0.6rem',
                  fontSize: '0.8rem',
                  backgroundColor: r.ppcLoading ? '#ccc' : '#2196F3',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: r.ppcLoading ? 'not-allowed' : 'pointer',
                }}
              >
                {r.ppcLoading ? 'Computing...' : 'Recompute PPC'}
              </button>
            </div>

            <div style={{ marginTop: '0.6rem', padding: '0.6rem', backgroundColor: '#f0f8ff', borderRadius: '4px', border: '1px solid #b3d9ff' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <button
                  onClick={async () => {
                    if (!r.ppcData) return
                    const next = !r.showOverlayOnMain
                    setCapturedRegions((prev) => prev.map((x) => (x.id === r.id ? { ...x, showOverlayOnMain: next } : x)))
                    if (next && !r.overlayDataUrl) {
                      try {
                        await ensureMainOverlayDataUrl(r)
                      } catch (e) {
                        console.error(e)
                        setCapturedRegions((prev) => prev.map((x) => (x.id === r.id ? { ...x, showOverlayOnMain: false } : x)))
                        alert(String(e))
                      }
                    }
                  }}
                  disabled={!r.ppcData}
                  style={{
                    padding: '0.45rem 0.6rem',
                    fontSize: '0.8rem',
                    backgroundColor: r.showOverlayOnMain ? '#4CAF50' : '#2196F3',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: r.ppcData ? 'pointer' : 'not-allowed',
                    opacity: r.ppcData ? 1 : 0.6,
                    fontWeight: 600,
                  }}
                >
                  {r.showOverlayOnMain ? '✓ Hide Overlay on Main Viewer' : 'Show Overlay on Main Viewer'}
                </button>

                <button
                  onClick={() => {
                    if (!r.ppcData) return
                    setCapturedRegions((prev) => prev.map((x) => (
                      x.id === r.id ? { ...x, showOverlayOnThumbnail: !x.showOverlayOnThumbnail } : x
                    )))
                  }}
                  disabled={!r.ppcData}
                  style={{
                    padding: '0.45rem 0.6rem',
                    fontSize: '0.8rem',
                    backgroundColor: r.showOverlayOnThumbnail ? '#4CAF50' : '#2196F3',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: r.ppcData ? 'pointer' : 'not-allowed',
                    opacity: r.ppcData ? 1 : 0.6,
                    fontWeight: 600,
                  }}
                >
                  {r.showOverlayOnThumbnail ? '✓ Hide Overlay on Thumbnail' : 'Show Overlay on Thumbnail'}
                </button>

                {(r.showOverlayOnMain || r.showOverlayOnThumbnail) && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem' }}>
                    <span style={{ color: '#666', minWidth: '60px' }}>Opacity:</span>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={r.overlayOpacity}
                      onChange={(e) => {
                        const v = parseFloat(e.target.value)
                        setCapturedRegions((prev) => prev.map((x) => (x.id === r.id ? { ...x, overlayOpacity: v } : x)))
                      }}
                      style={{ flex: 1 }}
                    />
                    <span style={{ color: '#666', minWidth: '45px', fontWeight: 600 }}>{(r.overlayOpacity * 100).toFixed(0)}%</span>
                  </div>
                )}
              </div>
            </div>

            {r.ppcData && (
              <div style={{ marginTop: '0.6rem' }}>
                <RegionPpcResults regionPpcData={r.ppcData} />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
