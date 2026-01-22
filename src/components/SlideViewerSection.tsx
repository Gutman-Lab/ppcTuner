import { useCallback, useState, useEffect, useMemo } from 'react'
import type { Dispatch, SetStateAction } from 'react'
import { SlideViewer } from 'bdsa-react-components'
import type { SlideViewerProps } from 'bdsa-react-components'
import { safeJsonParse } from '../utils/api'

// OverlayTileSource is exported from SlideViewer but not from main package export
// Import it from the component's types
type OverlayTileSource = NonNullable<SlideViewerProps['overlayTileSources']>[number]
import type { PpcResult, SavedParameterSet, CapturedRegion, RoiHueSample, RoiHueSampleDebug } from '../types'

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
  setSavedParameterSets: Dispatch<SetStateAction<SavedParameterSet[]>>
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
  setShowTissueMask: (show: boolean) => void
  histogramData: any
  // ROI sampling overlay (debug/validation)
  sampledRois: RoiHueSample[]
  sampledRoiDebug?: RoiHueSampleDebug[]
  showSampledRois: boolean
  setShowSampledRois: (show: boolean) => void
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
  setShowTissueMask,
  histogramData,
  sampledRois,
  sampledRoiDebug,
  showSampledRois,
  setShowSampledRois,
}: SlideViewerSectionProps) {
  // State for base64 data URLs (OpenSeadragon needs base64 data URLs for regular images in overlays)
  const [ppcLabelImageDataUrl, setPpcLabelImageDataUrl] = useState<string | null>(null)
  const [savedOverlayDataUrls, setSavedOverlayDataUrls] = useState<Record<string, string | null>>({})
  const [tissueMaskDataUrl, setTissueMaskDataUrl] = useState<string | null>(null)
  const [isCapturingRegion, setIsCapturingRegion] = useState(false)
  const [slideLoadError, setSlideLoadError] = useState<string | null>(null)
  
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

  // Fetch and convert tissue mask to base64 data URL so it can be added as an OSD overlay.
  useEffect(() => {
    if (!showTissueMask) {
      setTissueMaskDataUrl(null)
      return
    }
    const backgroundThreshold = histogramData?.tissue_analysis?.background_threshold ?? 240
    const maskUrl = `/api/images/${itemId}/mask?width=2048&background_threshold=${backgroundThreshold}`

    let cancelled = false
    fetch(maskUrl)
      .then((response) => {
        if (!response.ok) throw new Error(`Failed to load mask: ${response.status}`)
        return response.blob()
      })
      .then((blob) => {
        const reader = new FileReader()
        reader.onloadend = () => {
          if (!cancelled) setTissueMaskDataUrl(reader.result as string)
        }
        reader.onerror = () => {
          console.error('Error converting tissue mask to base64')
          if (!cancelled) setTissueMaskDataUrl(null)
        }
        reader.readAsDataURL(blob)
      })
      .catch((e) => {
        console.error('Error loading tissue mask overlay:', e)
        if (!cancelled) setTissueMaskDataUrl(null)
      })

    return () => {
      cancelled = true
    }
  }, [showTissueMask, itemId, histogramData])

  // Fetch and cache saved parameter-set overlays (full-image overlays)
  useEffect(() => {
    const visibleSets = savedParameterSets.filter((s) => s.visible && (!s.itemId || s.itemId === itemId))
    if (visibleSets.length === 0) {
      // Nothing visible; avoid setState churn (which can cause render loops).
      // If we already have cached overlays, drop them to prevent memory bloat.
      setSavedOverlayDataUrls((prev) => {
        if (Object.keys(prev).length === 0) return prev
        return {}
      })
      return
    }

    let cancelled = false
    const toFetch = visibleSets.filter((s) => !savedOverlayDataUrls[s.id])
    if (toFetch.length === 0) return () => { cancelled = true }

    ;(async () => {
      for (const s of toFetch) {
        try {
          const params = new URLSearchParams({
            item_id: itemId,
            method: 'hsi',
            thumbnail_width: '2048',
            hue_value: s.hueValue.toString(),
            hue_width: s.hueWidth.toString(),
            saturation_minimum: s.saturationMinimum.toString(),
            intensity_upper_limit: s.intensityUpperLimit.toString(),
            intensity_weak_threshold: s.intensityWeakThreshold.toString(),
            intensity_strong_threshold: s.intensityStrongThreshold.toString(),
            intensity_lower_limit: s.intensityLowerLimit.toString(),
            show_weak: showWeak.toString(),
            show_plain: showPlain.toString(),
            show_strong: showStrong.toString(),
            color_scheme: labelColorScheme,
          })

          const url = `/api/ppc/label-image?${params.toString()}`
          const response = await fetch(url)
          if (!response.ok) throw new Error(`Failed to load saved overlay: ${response.status}`)
          const blob = await response.blob()
          const dataUrl = await new Promise<string>((resolve, reject) => {
            const reader = new FileReader()
            reader.onloadend = () => resolve(reader.result as string)
            reader.onerror = () => reject(new Error('Failed to convert saved overlay to base64'))
            reader.readAsDataURL(blob)
          })

          if (cancelled) return
          setSavedOverlayDataUrls((prev) => ({ ...prev, [s.id]: dataUrl }))
        } catch (e) {
          // Don’t spam alerts; log once and keep going
          console.error('Failed to load saved overlay:', s.id, e)
          if (cancelled) return
          setSavedOverlayDataUrls((prev) => ({ ...prev, [s.id]: null }))
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [savedParameterSets, itemId, savedOverlayDataUrls, showWeak, showPlain, showStrong, labelColorScheme])
  
  
  const sampledRoiOverlays = useMemo(() => {
    if (!showSampledRois) return [] as OverlayTileSource[]

    const debug = sampledRoiDebug?.length ? sampledRoiDebug : null
    const accepted: Array<{ x: number, y: number, width: number, height: number }> =
      sampledRois.map((r) => ({ x: r.x, y: r.y, width: r.width, height: r.height }))

    const items: Array<{ status: 'accepted' | 'rejected', label: string, color: string, x: number, y: number, width: number, height: number }> = []

    if (debug) {
      let okIdx = 0
      let badIdx = 0
      for (const d of debug) {
        const color = d.status === 'accepted' ? '#00c853' : '#f44336'
        const idx = d.status === 'accepted' ? (++okIdx) : (++badIdx)
        const reasonKey = d.status === 'rejected' ? (d.reason ?? 'rejected') : 'ok'
        const shortReason =
          reasonKey === 'dab_band_fraction_thumb' ? 'strong_dab' :
          reasonKey === 'tissue_fraction' ? 'tissue' :
          reasonKey === 'too_close' ? 'close' :
          reasonKey === 'region_fetch_failed' ? 'fetch' :
          reasonKey === 'likely_negative_no_dab' ? 'neg' :
          String(reasonKey)

        const dabFrac = typeof d.dab_band_fraction_thumb === 'number' ? d.dab_band_fraction_thumb : null
        const label = d.status === 'accepted'
          ? `OK ${idx}`
          : `BAD ${idx}: ${shortReason}${dabFrac != null && shortReason === 'strong_dab' ? ` ${(dabFrac * 100).toFixed(1)}%` : ''}`
        items.push({ status: d.status, label, color, x: d.x, y: d.y, width: d.width, height: d.height })
      }
    } else {
      const colors = ['#ff00aa', '#00bcd4', '#ff9800', '#8bc34a', '#673ab7', '#e91e63', '#03a9f4']
      accepted.forEach((r, idx) => {
        items.push({
          status: 'accepted',
          label: `ROI ${idx + 1}`,
          color: colors[idx % colors.length],
          ...r,
        })
      })
    }

    if (items.length === 0) return [] as OverlayTileSource[]

    const makeSvgDataUrl = (label: string, color: string) => {
      const svg = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect x="2" y="2" width="96" height="96" fill="none" stroke="${color}" stroke-width="4"/>
  <rect x="2" y="2" width="96" height="20" fill="rgba(255,255,255,0.85)"/>
  <text x="6" y="16" font-family="Arial, sans-serif" font-size="12" fill="${color}" font-weight="700">${label}</text>
</svg>`
      return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
    }

    return items.map((r, idx) => {
      return {
        id: `roi-sampled-${idx + 1}-${r.status}`,
        tileSource: makeSvgDataUrl(r.label, r.color),
        x: r.x,
        y: r.y,
        width: r.width,
        height: r.height,
        opacity: 1.0,
      }
    })
  }, [sampledRois, sampledRoiDebug, showSampledRois])

  // Combine all overlays (PPC overlays + multiple region overlays + sampled ROI boxes)
  const overlayTileSources = useMemo(() => {
    const overlays: OverlayTileSource[] = []

    if (showTissueMask && tissueMaskDataUrl) {
      overlays.push({
        id: 'tissue-mask',
        tileSource: tissueMaskDataUrl,
        x: 0,
        y: 0,
        width: 1,
        height: 1,
        opacity: 0.45,
      })
    }

    for (const o of sampledRoiOverlays) {
      overlays.push(o)
    }

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

    for (const s of savedParameterSets) {
      if (!s.visible) continue
      if (s.itemId && s.itemId !== itemId) continue
      const tileSource = savedOverlayDataUrls[s.id]
      if (!tileSource) continue
      overlays.push({
        id: `ppc-label-saved-${s.id}`,
        tileSource,
        x: 0,
        y: 0,
        width: 1,
        height: 1,
        opacity: ppcLabelOpacity * 0.7,
      })
    }

    return overlays
  }, [showTissueMask, tissueMaskDataUrl, sampledRoiOverlays, showPpcLabel, ppcLabelImageDataUrl, ppcLabelOpacity, capturedRegions, savedParameterSets, savedOverlayDataUrls, itemId])
  
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
      
      const response = await fetch(finalUrl, {
        ...options,
        headers,
      })
      
      // Capture common failure modes so we don't end up with a silent black viewer.
      // (OSD can fail quietly depending on where the request fails.)
      if (!response.ok) {
        const isTilesRequest = finalUrl.includes('/tiles/')
        if (isTilesRequest) {
          const msg = `Slide load failed (${response.status})${response.status === 401 ? ' — token/auth?' : ''}`
          setSlideLoadError((prev) => prev ?? msg)
        }
      } else {
        // Clear any previous error once we get a successful tiles request.
        if (finalUrl.includes('/tiles/')) {
          setSlideLoadError(null)
        }
      }
      
      return response
    },
    [dsaToken]
  )

  return (
    <div style={{ position: 'relative', display: 'inline-block', width: '100%' }}>
      <div style={{ width: '100%', height, position: 'relative' }}>
        {slideLoadError && (
          <div
            style={{
              position: 'absolute',
              left: '10px',
              bottom: '10px',
              zIndex: 1000,
              background: 'rgba(255,255,255,0.95)',
              border: '1px solid #f5a623',
              borderRadius: '6px',
              padding: '0.4rem 0.6rem',
              fontSize: '0.75rem',
              color: '#333',
              maxWidth: '420px',
              boxShadow: '0 2px 6px rgba(0,0,0,0.12)',
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: '0.15rem' }}>Viewer warning</div>
            <div>{slideLoadError}</div>
            {!dsaToken && (
              <div style={{ marginTop: '0.25rem', color: '#666' }}>
                No token in frontend; if your DSA requires auth, tiles will 401.
              </div>
            )}
          </div>
        )}
        <SlideViewer
          key={`${itemId}-${dsaToken ? 'auth' : 'noauth'}-overlays-${showPpcLabel ? '1' : '0'}-mask-${showTissueMask ? '1' : '0'}-rois-${showSampledRois ? '1' : '0'}-${(sampledRoiDebug?.length ?? sampledRois.length)}-regions-${capturedRegions.filter(r => r.showOverlayOnMain).length}`} // Force re-render when overlay visibility changes to prevent ghost overlays
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
              
              // Keep this log concise; it's useful when debugging overlay alignment issues.
              console.log(
                'Captured region:',
                `x=${clampedRegion.x.toFixed(4)}`,
                `y=${clampedRegion.y.toFixed(4)}`,
                `w=${clampedRegion.width.toFixed(4)}`,
                `h=${clampedRegion.height.toFixed(4)}`
              )
              
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

            {showTissueMask && !!tissueMaskDataUrl && (
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.5rem' }}>
                <div style={{ color: '#333' }}>Tissue Mask</div>
                <button
                  onClick={() => setShowTissueMask(false)}
                  style={{
                    padding: '0.15rem 0.5rem',
                    fontSize: '0.7rem',
                    backgroundColor: '#f44336',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                  }}
                  title="Remove tissue mask overlay"
                >
                  Remove
                </button>
              </div>
            )}

            {showSampledRois && sampledRois.length > 0 && (
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.5rem' }}>
                <div style={{ color: '#333' }}>Sampled ROIs</div>
                <button
                  onClick={() => setShowSampledRois(false)}
                  style={{
                    padding: '0.15rem 0.5rem',
                    fontSize: '0.7rem',
                    backgroundColor: '#f44336',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                  }}
                  title="Hide sampled ROI boxes"
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
              (showTissueMask && !!tissueMaskDataUrl) ||
              (showSampledRois && sampledRois.length > 0) ||
              (showPpcLabel && !!ppcLabelImageDataUrl) ||
              capturedRegions.some((r) => r.showOverlayOnMain && !!r.overlayDataUrl) ||
              savedParameterSets.some((s) => s.visible && (!s.itemId || s.itemId === itemId))
            ) && <div style={{ color: '#777', fontStyle: 'italic' }}>None</div>}
          </div>
        </div>
      </div>
    </div>
  )
}
