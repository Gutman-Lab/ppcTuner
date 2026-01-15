import { useCallback } from 'react'
import { safeJsonParse } from '../utils/api'

interface UseAutoDetectHueProps {
  setHueValue: (value: number) => void
  setHueWidth: (value: number) => void
  setSaturationMinimum: (value: number) => void
  setIntensityUpperLimit: (value: number) => void
  setIntensityWeakThreshold: (value: number) => void
  setIntensityStrongThreshold: (value: number) => void
  setIntensityLowerLimit: (value: number) => void
}

export function useAutoDetectHue({
  setHueValue,
  setHueWidth,
  setSaturationMinimum,
  setIntensityUpperLimit,
  setIntensityWeakThreshold,
  setIntensityStrongThreshold,
  setIntensityLowerLimit,
}: UseAutoDetectHueProps) {
  const autoDetectHue = useCallback(async (itemId: string) => {
    try {
      console.log('🔍 Auto-detecting hue parameters for item:', itemId)
      const response = await fetch(`/api/ppc/auto-detect-hue?item_id=${itemId}&thumbnail_width=1024`)
      if (response.ok) {
        const data = await safeJsonParse(response)
        setHueValue(data.hue_value)
        setHueWidth(data.hue_width)
        setSaturationMinimum(data.saturation_minimum)
        setIntensityUpperLimit(data.intensity_upper_limit)
        setIntensityWeakThreshold(data.intensity_weak_threshold)
        setIntensityStrongThreshold(data.intensity_strong_threshold)
        setIntensityLowerLimit(data.intensity_lower_limit)
        console.log('✓ Auto-detected hue parameters:', data)
      } else {
        console.error('Auto-detect failed:', response.status)
      }
    } catch (error) {
      console.error('Error auto-detecting hue:', error)
    }
  }, [
    setHueValue,
    setHueWidth,
    setSaturationMinimum,
    setIntensityUpperLimit,
    setIntensityWeakThreshold,
    setIntensityStrongThreshold,
    setIntensityLowerLimit,
  ])

  return { autoDetectHue }
}
