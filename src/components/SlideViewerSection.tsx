import { useCallback, useState, useEffect, useMemo } from 'react'
import type { Dispatch, SetStateAction } from 'react'
import { SlideViewer } from 'bdsa-react-components'
import type { SlideViewerProps } from 'bdsa-react-components'
import { TissueMaskOverlay } from './TissueMaskOverlay'
import { safeJsonParse } from '../utils/api'

// OverlayTileSource is exported from SlideViewer but not from main package export
// Import it from the component's types
type OverlayTileSource = NonNullable<SlideViewerProps['overlayTileSources']>[number]
import type { PpcResult, SavedParameterSet, CapturedRegion } from '../types'

interface SlideViewerSectionProps {
  itemId: string
  apiBaseUrl: string
  dsaToken: string | null
  height?: string
  // Viewport tracking
  currentViewport: { x: number, y: number, width: number, height: number, zoom: number } | null
  setCurrentViewport: (viewport: { x: number, y: number, width: number, height: number, zoom: number } | null) => void
  // Region capture (multiple)
  capturedRegions: CapturedRegion[]
  setCapturedRegions: Dispatch<SetStateAction<CapturedRegion[]>>
  // PPC overlays
  showPpcLabel: boolean
  setShowPpcLabel: (show: boolean) => void
  ppcData: PpcResult | null
  ppcLabelOpacity: number
  showWeak: boolean
  showPlain: boolean
  showStrong: boolean
  labelColorScheme: string
  // Saved parameter set overlays
  savedParameterSets: SavedParameterSet[]
  setSavedParameterSets: (sets: SavedParameterSet[]) => void
  // HSI parameters
  hueValue: number
  hueWidth: number
  saturationMinimum: number
  intensityUpperLimit: number
  intensityWeakThreshold: number
  intensityStrongThreshold: number
  intensityLowerLimit: number
  // Tissue mask
  showTissueMask: boolean
  histogramData: any
}

export function SlideViewerSection({
  itemId,
  apiBaseUrl,
  dsaToken,
  height = '600px',
  currentViewport,
  setCurrentViewport,
  capturedRegions,
  setCapturedRegions,
  showPpcLabel,
  setShowPpcLabel,
  ppcData,
  ppcLabelOpacity,
  showWeak,
  showPlain,
  showStrong,
  labelColorScheme,
  savedParameterSets,
  setSavedParameterSets,
  hueValue,
  hueWidth,
  saturationMinimum,
  intensityUpperLimit,
  intensityWeakThreshold,
  intensityStrongThreshold,
  intensityLowerLimit,
  showTissueMask,
  histogramData,
}: SlideViewerSectionProps) {
  // State for base64 data URLs (OpenSeadragon needs base64 data URLs for regular images in overlays)
  const [ppcLabelImageDataUrl, setPpcLabelImageDataUrl] = useState<string | null>(null)
  const [isCapturingRegion, setIsCapturingRegion] = useState(false)
  
  // Build PPC label image URLs (served via backend proxy - more efficient than base64)
  // The overlayTileSources prop accepts image URLs directly
  const ppcLabelImageUrl = useMemo(() => {
    if (!showPpcLabel || ppcData?.method !== 'hsi' || !itemId) return null
    
    const params = new URLSearchParams({
      item_id: itemId,
      method: 'hsi',
      thumbnail_width: '2048',
      hue_value: hueValue.toString(),
      hue_width: hueWidth.toString(),
      saturation_minimum: saturationMinimum.toString(),
      intensity_upper_limit: intensityUpperLimit.toString(),
      intensity_weak_threshold: intensityWeakThreshold.toString(),
      intensity_strong_threshold: intensityStrongThreshold.toString(),
      intensity_lower_limit: intensityLowerLimit.toString(),
      show_weak: showWeak.toString(),
      show_plain: showPlain.toString(),
      show_strong: showStrong.toString(),
      color_scheme: labelColorScheme,
    })
    
    return `/api/ppc/label-image?${params.toString()}`
  }, [
    showPpcLabel,
    ppcData,
    itemId,
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
  
  // Fetch and convert PPC label image to base64 data URL
  useEffect(() => {
    if (!ppcLabelImageUrl) {
      setPpcLabelImageDataUrl(null)
      return
    }
    
    fetch(ppcLabelImageUrl)
      .then(response => {
        if (!response.ok) throw new Error(`Failed to load PPC label: ${response.status}`)
        return response.blob()
      })
      .then(blob => {
        // Convert blob to base64 data URL
        const reader = new FileReader()
        reader.onloadend = () => {
          setPpcLabelImageDataUrl(reader.result as string)
        }
        reader.onerror = () => {
          console.error('Error converting PPC label to base64')
          setPpcLabelImageDataUrl(null)
        }
        reader.readAsDataURL(blob)
      })
      .catch(error => {
        console.error('Error loading PPC label overlay:', error)
        setPpcLabelImageDataUrl(null)
      })
  }, [ppcLabelImageUrl])
  
  
  // Combine all overlays (PPC overlays + multiple region overlays)
  const overlayTileSources = useMemo(() => {
    const overlays: OverlayTileSource[] = []

    if (showPpcLabel && ppcLabelImageDataUrl) {
      overlays.push({
        id: 'ppc-label-full',
        tileSource: ppcLabelImageDataUrl,
        x: 0,
        y: 0,
        width: 1,
        height: 1,
        opacity: ppcLabelOpacity,
      })
    }

    for (const r of capturedRegions) {
      if (!r.showOverlayOnMain) continue
      if (!r.overlayDataUrl) continue
      overlays.push({
        id: `ppc-label-region-${r.id}`,
        tileSource: r.overlayDataUrl,
        x: r.region.x,
        y: r.region.y,
        width: r.region.width,
        height: r.region.height,
        opacity: r.overlayOpacity,
      })
    }

    return overlays
  }, [showPpcLabel, ppcLabelImageDataUrl, ppcLabelOpacity, capturedRegions])
  
  // Memoize fetchFn for SlideViewer to ensure proper authentication
  // Always provide fetchFn (even if no token) to ensure it's used for all requests
  const slideViewerFetchFn = useCallback(
    async (url: string, options?: RequestInit) => {
      // Always add token to headers if available
      const headers: Record<string, string> = {
        ...(options?.headers as Record<string, string> || {}),
      }
      
      if (dsaToken) {
        headers['Girder-Token'] = dsaToken
      }
      
      // Add token to query string for OpenSeadragon tile requests
      let finalUrl = url
      if (dsaToken) {
        try {
          const urlObj = new URL(url, window.location.origin)
          if (!urlObj.searchParams.has('token')) {
            urlObj.searchParams.set('token', dsaToken)
          }
          finalUrl = urlObj.toString()
        } catch (error) {
          // If URL parsing fails (e.g., relative URL), use original URL
          // Token will still be in headers
          console.warn('Failed to parse URL for token injection, using headers only:', error)
        }
      }
      
      console.log('SlideViewer fetch:', finalUrl)
      console.log('SlideViewer has token:', !!dsaToken)
      console.log('SlideViewer headers:', headers)
      
      const response = await fetch(finalUrl, {
        ...options,
        headers,
      })
      
      console.log('SlideViewer response status:', response.status, 'for', finalUrl)
      
      if (!response.ok && response.status === 401) {
        console.error('SlideViewer 401 Unauthorized - token may be invalid or expired')
        console.error('Request URL:', finalUrl)
        console.error('Request headers:', headers)
        const errorText = await response.clone().text()
        console.error('Error response:', errorText.substring(0, 200))
      }
      
      return response
    },
    [dsaToken]
  )

  return (
    <div style={{ position: 'relative', display: 'inline-block', width: '100%' }}>
      <div style={{ width: '100%', height, position: 'relative' }}>
        <SlideViewer
          key={`${itemId}-${dsaToken ? 'auth' : 'noauth'}-overlays-${showPpcLabel ? '1' : '0'}-regions-${capturedRegions.filter(r => r.showOverlayOnMain).length}`} // Force re-render when overlay visibility changes to prevent ghost overlays
          imageInfo={{
            dziUrl: dsaToken 
              ? `${apiBaseUrl}/item/${itemId}/tiles/dzi.dzi?token=${dsaToken}`
              : `${apiBaseUrl}/item/${itemId}/tiles/dzi.dzi`
          }}
          height={height}
          apiBaseUrl={apiBaseUrl}
          apiHeaders={dsaToken ? { 'Girder-Token': dsaToken } : undefined}
          fetchFn={slideViewerFetchFn}
          overlayTileSources={overlayTileSources}
          onViewportChange={(bounds: { x: number, y: number, width: number, height: number, zoom: number }) => {
            // Use the viewport coordinates from SlideViewer
            // Note: Coordinates are normalized (0-1) relative to full image
            setCurrentViewport({
              x: bounds.x,
              y: bounds.y,
              width: bounds.width,
              height: bounds.height,
              zoom: bounds.zoom
            })
          }}
        />
        {/* Capture Viewport Region Button */}
        <button
          onClick={async () => {
            // Use the current viewport state instead of accessing viewer directly
            if (!currentViewport) {
              alert('Viewport not available yet. Please wait for the viewer to load and display the image.')
              return
            }
            
            // Validate viewport coordinates
            if (
              typeof currentViewport.x !== 'number' ||
              typeof currentViewport.y !== 'number' ||
              typeof currentViewport.width !== 'number' ||
              typeof currentViewport.height !== 'number' ||
              currentViewport.width <= 0 ||
              currentViewport.height <= 0
            ) {
              alert('Invalid viewport coordinates. Please wait for the viewer to fully load.')
              return
            }
            
            try {
              // Use the tracked viewport coordinates
              // CRITICAL: These normalized coordinates (0-1) must be preserved exactly
              // for the overlay to align correctly with the captured region
              // However, we need to validate and clamp coordinates to valid range (0-1)
              // because before zooming in, coordinates might be negative or > 1
              const region = {
                x: currentViewport.x,
                y: currentViewport.y,
                width: currentViewport.width,
                height: currentViewport.height
              }
              
              // Clamp coordinates to valid range (0-1) for normalized coordinates.
              // When zoomed out, x/y can be negative; snap negatives to 0 so we can still capture a partial FOV.
              const clampedRegion = {
                x: Math.max(0, region.x),
                y: Math.max(0, region.y),
                // Width/height should never be negative, but if they are, snap to 0 (we'll validate below).
                width: Math.max(0, region.width),
                height: Math.max(0, region.height)
              }
              
              // Ensure the region doesn't extend beyond the image bounds
              if (clampedRegion.x + clampedRegion.width > 1) {
                clampedRegion.width = 1 - clampedRegion.x
              }
              if (clampedRegion.y + clampedRegion.height > 1) {
                clampedRegion.height = 1 - clampedRegion.y
              }
              
              // Validate final coordinates are meaningful (not too small)
              if (clampedRegion.width <= 0 || clampedRegion.height <= 0) {
                alert('Error: Captured region is too small. Please zoom in and try again.')
                return
              }
              // Avoid absurdly tiny regions due to rounding/clamping
              if (clampedRegion.width < 0.001 || clampedRegion.height < 0.001) {
                alert('Error: Captured region is too small. Please zoom in and try again.')
                return
              }
              
              console.log('Captured viewport region (normalized 0-1):', clampedRegion)
              console.log('Region coordinates - X:', clampedRegion.x.toFixed(6), 'Y:', clampedRegion.y.toFixed(6), 
                         'W:', clampedRegion.width.toFixed(6), 'H:', clampedRegion.height.toFixed(6))
              
              const regionId = `region-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`
              const createdAt = Date.now()
              const paramsUsed = {
                hueValue,
                hueWidth,
                saturationMinimum,
                intensityUpperLimit,
                intensityWeakThreshold,
                intensityStrongThreshold,
                intensityLowerLimit,
              }
              const displayUsed = {
                showWeak,
                showPlain,
                showStrong,
                labelColorScheme,
              }

              setCapturedRegions((prev) => ([
                ...prev,
                {
                  id: regionId,
                  itemId,
                  createdAt,
                  region: clampedRegion,
                  imageUrl: null,
                  isCapturing: true,
                  ppcData: null,
                  ppcLoading: true,
                  paramsUsed,
                  displayUsed,
                  showOverlayOnMain: false,
                  showOverlayOnThumbnail: false,
                  overlayOpacity: 0.5,
                  overlayDataUrl: null,
                },
              ]))

              setIsCapturingRegion(true)
              
              // Fetch cropped region from backend (use clamped coordinates)
              const params = new URLSearchParams({
                x: clampedRegion.x.toString(),
                y: clampedRegion.y.toString(),
                width: clampedRegion.width.toString(),
                height: clampedRegion.height.toString(),
                output_width: '1024'
              })

              const regionImgPromise = fetch(`/api/images/${itemId}/region?${params.toString()}`)
                .then(async (response) => {
                  if (!response.ok) {
                    const errorText = await response.text()
                    throw new Error(`Failed to fetch region: ${response.status} ${errorText}`)
                  }
                  const blob = await response.blob()
                  return URL.createObjectURL(blob)
                })

              const ppcParams = new URLSearchParams({
                item_id: itemId,
                x: clampedRegion.x.toString(),
                y: clampedRegion.y.toString(),
                width: clampedRegion.width.toString(),
                height: clampedRegion.height.toString(),
                output_width: '1024',
                hue_value: paramsUsed.hueValue.toString(),
                hue_width: paramsUsed.hueWidth.toString(),
                saturation_minimum: paramsUsed.saturationMinimum.toString(),
                intensity_upper_limit: paramsUsed.intensityUpperLimit.toString(),
                intensity_weak_threshold: paramsUsed.intensityWeakThreshold.toString(),
                intensity_strong_threshold: paramsUsed.intensityStrongThreshold.toString(),
                intensity_lower_limit: paramsUsed.intensityLowerLimit.toString(),
              })

              const ppcPromise = fetch(`/api/ppc/compute-region?${ppcParams.toString()}`)
                .then(async (response) => {
                  if (!response.ok) {
                    const errorText = await response.text()
                    throw new Error(`Failed to compute PPC on region: ${response.status} ${errorText}`)
                  }
                  return safeJsonParse(response)
                })

              const [imgRes, ppcRes] = await Promise.allSettled([regionImgPromise, ppcPromise])
              setCapturedRegions((prev) => prev.map((r) => {
                if (r.id !== regionId) return r
                return {
                  ...r,
                  imageUrl: imgRes.status === 'fulfilled' ? imgRes.value : r.imageUrl,
                  isCapturing: false,
                  ppcData: ppcRes.status === 'fulfilled' ? (ppcRes.value as any) : r.ppcData,
                  ppcLoading: false,
                }
              }))

              if (imgRes.status === 'rejected') console.error(imgRes.reason)
              if (ppcRes.status === 'rejected') console.error(ppcRes.reason)
            } catch (error) {
              console.error('Error capturing viewport:', error)
              alert(`Error: ${error}`)
            } finally {
              setIsCapturingRegion(false)
            }
          }}
          disabled={!currentViewport || isCapturingRegion}
          style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            padding: '0.5rem 1rem',
            backgroundColor: (!currentViewport || isCapturingRegion) ? '#ccc' : '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: (!currentViewport || isCapturingRegion) ? 'not-allowed' : 'pointer',
            fontSize: '0.875rem',
            zIndex: 1000,
            boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
            opacity: (!currentViewport || isCapturingRegion) ? 0.6 : 1
          }}
          title={!currentViewport ? 'Waiting for viewer to load...' : 'Capture the currently visible region'}
        >
          {isCapturingRegion ? 'Capturing...' : 'Capture Viewport Region'}
        </button>

        {/* Simple overlay registry (reflects what we are currently adding to OSD via overlayTileSources) */}
        <div
          style={{
            position: 'absolute',
            left: '10px',
            top: '10px',
            zIndex: 1000,
            background: 'rgba(255,255,255,0.92)',
            border: '1px solid #ddd',
            borderRadius: '6px',
            padding: '0.5rem 0.6rem',
            fontSize: '0.75rem',
            minWidth: '220px',
            boxShadow: '0 2px 6px rgba(0,0,0,0.12)',
          }}
        >
          <div style={{ fontWeight: 700, color: '#333', marginBottom: '0.35rem' }}>
            Active Overlays
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            {showPpcLabel && !!ppcLabelImageDataUrl && (
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.5rem' }}>
                <div style={{ color: '#333' }}>Full PPC</div>
                <button
                  onClick={() => setShowPpcLabel(false)}
                  style={{
                    padding: '0.15rem 0.5rem',
                    fontSize: '0.7rem',
                    backgroundColor: '#f44336',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                  }}
                  title="Remove full-image overlay"
                >
                  Remove
                </button>
              </div>
            )}

            {capturedRegions
              .filter((r) => r.showOverlayOnMain && !!r.overlayDataUrl)
              .map((r, idx) => (
                <div
                  key={r.id}
                  style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.5rem' }}
                >
                  <div style={{ color: '#333' }}>Region PPC #{idx + 1}</div>
                  <button
                    onClick={() => {
                      setCapturedRegions(capturedRegions.map((x) => (
                        x.id === r.id ? { ...x, showOverlayOnMain: false } : x
                      )))
                    }}
                    style={{
                      padding: '0.15rem 0.5rem',
                      fontSize: '0.7rem',
                      backgroundColor: '#f44336',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                    }}
                    title="Remove region overlay"
                  >
                    Remove
                  </button>
                </div>
              ))}

            {savedParameterSets
              .filter((s) => s.visible && (!s.itemId || s.itemId === itemId))
              .map((s) => (
                <div
                  key={s.id}
                  style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.5rem' }}
                >
                  <div style={{ color: '#333', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    Saved: {s.name}
                  </div>
                  <button
                    onClick={() => {
                      setSavedParameterSets(savedParameterSets.map((x) => (x.id === s.id ? { ...x, visible: false } : x)))
                    }}
                    style={{
                      padding: '0.15rem 0.5rem',
                      fontSize: '0.7rem',
                      backgroundColor: '#f44336',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      flex: '0 0 auto',
                    }}
                    title="Remove saved overlay (hides it)"
                  >
                    Remove
                  </button>
                </div>
              ))}

            {!(
              (showPpcLabel && !!ppcLabelImageDataUrl) ||
              capturedRegions.some((r) => r.showOverlayOnMain && !!r.overlayDataUrl) ||
              savedParameterSets.some((s) => s.visible && (!s.itemId || s.itemId === itemId))
            ) && <div style={{ color: '#777', fontStyle: 'italic' }}>None</div>}
          </div>
        </div>
      </div>
      {showTissueMask && histogramData?.tissue_analysis && (
        <TissueMaskOverlay 
          itemId={itemId}
          backgroundThreshold={histogramData.tissue_analysis.background_threshold || 240}
        />
      )}
    </div>
  )
}
