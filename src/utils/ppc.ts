/**
 * PPC (Positive Pixel Count) utility functions
 */
import type { PpcResult } from '../types'
import { round } from './api'

/**
 * Classify intensities in real-time (client-side, no API call needed)
 * This function reclassifies already-detected positive pixels based on intensity thresholds
 */
export function classifyIntensities(
  intensities: number[],
  tissuePixels: number,
  totalPixels: number,
  backgroundPixels: number,
  itemId: string,
  weakThreshold: number,
  strongThreshold: number,
  hueValue: number,
  hueWidth: number,
  saturationMinimum: number,
  intensityUpperLimit: number,
  intensityLowerLimit: number
): PpcResult {
  const weak = intensities.filter(i => i >= weakThreshold).length
  const strong = intensities.filter(i => i < strongThreshold).length
  const plain = intensities.length - weak - strong
  const totalPositive = intensities.length

  return {
    item_id: itemId,
    total_pixels: totalPixels,
    tissue_pixels: tissuePixels,
    background_pixels: backgroundPixels,
    method: 'hsi',
    weak_positive_pixels: weak,
    plain_positive_pixels: plain,
    strong_positive_pixels: strong,
    total_positive_pixels: totalPositive,
    weak_percentage: round((weak / tissuePixels * 100), 2),
    plain_percentage: round((plain / tissuePixels * 100), 2),
    strong_percentage: round((strong / tissuePixels * 100), 2),
    positive_percentage: round((totalPositive / tissuePixels * 100), 2),
    parameters: {
      hue_value: hueValue,
      hue_width: hueWidth,
      saturation_minimum: saturationMinimum,
      intensity_upper_limit: intensityUpperLimit,
      intensity_weak_threshold: weakThreshold,
      intensity_strong_threshold: strongThreshold,
      intensity_lower_limit: intensityLowerLimit,
      thumbnail_width: 1024,
    },
    metrics: {
      execution_time_seconds: 0.0, // Instant client-side classification
      cpu_percent: 0.0,
      memory_mb: 0.0,
    },
  }
}
