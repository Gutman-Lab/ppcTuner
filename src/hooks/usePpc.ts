/**
 * Custom hook for PPC computation and intensity map management
 */
import { useState, useCallback, useRef, useEffect } from 'react'
import type { PpcResult, IntensityMap } from '../types'
import { safeJsonParse } from '../utils/api'
import { classifyIntensities } from '../utils/ppc'

interface UsePpcParams {
  hueValue: number
  hueWidth: number
  saturationMinimum: number
  intensityUpperLimit: number
  intensityWeakThreshold: number
  intensityStrongThreshold: number
  intensityLowerLimit: number
  autoCompute: boolean
  selectedItem: any
}

export function usePpc({
  hueValue,
  hueWidth,
  saturationMinimum,
  intensityUpperLimit,
  intensityWeakThreshold,
  intensityStrongThreshold,
  intensityLowerLimit,
  autoCompute,
  selectedItem,
}: UsePpcParams) {
  const [ppcData, setPpcData] = useState<PpcResult | null>(null)
  const [ppcLoading, setPpcLoading] = useState(false)
  const [intensityMap, setIntensityMap] = useState<IntensityMap | null>(null)
  const [intensityMapLoading, setIntensityMapLoading] = useState<boolean>(false)
  const computeDebounceTimerRef = useRef<NodeJS.Timeout | null>(null)

  // Fetch intensity map (when hue/saturation/bounds change, not thresholds)
  const fetchIntensityMap = useCallback(
    async (itemId: string) => {
      if (!itemId) return

      setIntensityMapLoading(true)
      try {
        const url = `/api/ppc/intensity-map?item_id=${itemId}&thumbnail_width=1024&hue_value=${hueValue}&hue_width=${hueWidth}&saturation_minimum=${saturationMinimum}&intensity_upper_limit=${intensityUpperLimit}&intensity_lower_limit=${intensityLowerLimit}`
        const response = await fetch(url)
        if (response.ok) {
          const data = (await safeJsonParse(response)) as IntensityMap
          setIntensityMap(data)
          // Immediately classify with current thresholds
          if (data.positive_intensities.length > 0) {
            const classified = classifyIntensities(
              data.positive_intensities,
              data.tissue_pixels,
              data.total_pixels,
              data.background_pixels,
              itemId,
              intensityWeakThreshold,
              intensityStrongThreshold,
              hueValue,
              hueWidth,
              saturationMinimum,
              intensityUpperLimit,
              intensityLowerLimit
            )
            setPpcData(classified)
          } else {
            // No positive pixels
            const classified = classifyIntensities(
              [],
              data.tissue_pixels,
              data.total_pixels,
              data.background_pixels,
              itemId,
              intensityWeakThreshold,
              intensityStrongThreshold,
              hueValue,
              hueWidth,
              saturationMinimum,
              intensityUpperLimit,
              intensityLowerLimit
            )
            setPpcData(classified)
          }
        } else {
          console.error('Failed to fetch intensity map:', response.status)
        }
      } catch (error) {
        console.error('Error fetching intensity map:', error)
      } finally {
        setIntensityMapLoading(false)
      }
    },
    [
      hueValue,
      hueWidth,
      saturationMinimum,
      intensityUpperLimit,
      intensityLowerLimit,
      intensityWeakThreshold,
      intensityStrongThreshold,
    ]
  )

  // Reusable PPC computation function
  // Uses intensity map if available and only thresholds changed, otherwise full API call
  const computePpc = useCallback(
    async (itemId: string) => {
      if (!itemId) return

      // Check if we can use intensity map (only thresholds changed)
      if (intensityMap && intensityMap.item_id === itemId) {
        // We have an intensity map - classify in real-time (instant!)
        const classified = classifyIntensities(
          intensityMap.positive_intensities,
          intensityMap.tissue_pixels,
          intensityMap.total_pixels,
          intensityMap.background_pixels,
          itemId,
          intensityWeakThreshold,
          intensityStrongThreshold,
          hueValue,
          hueWidth,
          saturationMinimum,
          intensityUpperLimit,
          intensityLowerLimit
        )
        setPpcData(classified)
        return // No API call needed!
      }

      // Full API call needed (intensity map not available, or bounds changed)
      setPpcLoading(true)
      setPpcData(null)
      try {
        const url = `/api/ppc/compute?item_id=${itemId}&method=hsi&thumbnail_width=1024&hue_value=${hueValue}&hue_width=${hueWidth}&saturation_minimum=${saturationMinimum}&intensity_upper_limit=${intensityUpperLimit}&intensity_weak_threshold=${intensityWeakThreshold}&intensity_strong_threshold=${intensityStrongThreshold}&intensity_lower_limit=${intensityLowerLimit}`
        const response = await fetch(url)
        if (response.ok) {
          const data = await safeJsonParse(response)
          setPpcData(data)
        } else {
          const errorText = await response.text()
          console.error('PPC computation failed:', response.status, errorText)
          if (!autoCompute) {
            // Only show alert if manual compute
            alert(`PPC computation failed: ${response.status}`)
          }
        }
      } catch (error) {
        console.error('Error computing PPC:', error)
        if (!autoCompute) {
          // Only show alert if manual compute
          alert(`Error: ${error}`)
        }
      } finally {
        setPpcLoading(false)
      }
    },
    [
      hueValue,
      hueWidth,
      saturationMinimum,
      intensityUpperLimit,
      intensityWeakThreshold,
      intensityStrongThreshold,
      intensityLowerLimit,
      autoCompute,
      intensityMap,
    ]
  )

  // Fetch intensity map when hue/saturation/bounds change (not thresholds)
  // This allows instant threshold adjustment without API calls
  useEffect(() => {
    if (!autoCompute || !selectedItem) return

    const itemId = (selectedItem as any)._id || (selectedItem as any).id
    if (!itemId) return

    // Clear existing timer
    if (computeDebounceTimerRef.current) {
      clearTimeout(computeDebounceTimerRef.current)
    }

    // Debounce intensity map fetch (hue/saturation/bounds changed)
    computeDebounceTimerRef.current = setTimeout(() => {
      fetchIntensityMap(itemId)
    }, 500) // 500ms debounce

    return () => {
      if (computeDebounceTimerRef.current) {
        clearTimeout(computeDebounceTimerRef.current)
      }
    }
  }, [
    autoCompute,
    selectedItem,
    hueValue,
    hueWidth,
    saturationMinimum,
    intensityUpperLimit,
    intensityLowerLimit,
    fetchIntensityMap,
  ])

  // Real-time classification when thresholds change (if intensity map exists)
  useEffect(() => {
    if (!autoCompute || !selectedItem || !intensityMap) return

    const itemId = (selectedItem as any)._id || (selectedItem as any).id
    if (!itemId || intensityMap.item_id !== itemId) return

    // Instant reclassification (no API call needed!)
    const classified = classifyIntensities(
      intensityMap.positive_intensities,
      intensityMap.tissue_pixels,
      intensityMap.total_pixels,
      intensityMap.background_pixels,
      itemId,
      intensityWeakThreshold,
      intensityStrongThreshold,
      hueValue,
      hueWidth,
      saturationMinimum,
      intensityUpperLimit,
      intensityLowerLimit
    )
    setPpcData(classified)
  }, [
    autoCompute,
    selectedItem,
    intensityMap,
    intensityWeakThreshold,
    intensityStrongThreshold,
    hueValue,
    hueWidth,
    saturationMinimum,
    intensityUpperLimit,
    intensityLowerLimit,
  ])

  return {
    ppcData,
    setPpcData,
    ppcLoading,
    setPpcLoading,
    intensityMap,
    setIntensityMap,
    intensityMapLoading,
    computePpc,
    fetchIntensityMap,
  }
}
