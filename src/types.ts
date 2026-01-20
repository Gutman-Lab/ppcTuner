export interface AppConfig {
  dsaBaseUrl: string
  startFolder: string
  startFolderType: string
  dsaToken: string | null  // Token obtained from API key authentication
}

export interface PpcResult {
  item_id: string
  total_pixels: number
  tissue_pixels: number
  background_pixels: number
  method: string
  brown_pixels?: number
  yellow_pixels?: number
  red_pixels?: number
  total_positive_pixels?: number
  brown_percentage?: number
  yellow_percentage?: number
  red_percentage?: number
  positive_percentage?: number
  weak_positive_pixels?: number
  plain_positive_pixels?: number
  strong_positive_pixels?: number
  weak_percentage?: number
  plain_percentage?: number
  strong_percentage?: number
  parameters: { [key: string]: any }
  metrics: { [key: string]: number }
}

export interface IntensityMap {
  item_id: string
  positive_intensities: number[]  // Intensity values (0-1) for positive pixels
  positive_pixel_indices: number[][]  // [row, col] pairs for positive pixels
  tissue_pixels: number
  total_pixels: number
  background_pixels: number
  image_shape: [number, number]  // [height, width]
  metrics: { [key: string]: number }
}

export interface HsiParams {
  hueValue: number
  hueWidth: number
  saturationMinimum: number
  intensityUpperLimit: number
  intensityWeakThreshold: number
  intensityStrongThreshold: number
  intensityLowerLimit: number
}

export interface PpcDisplayOptions {
  showWeak: boolean
  showPlain: boolean
  showStrong: boolean
  labelColorScheme: string
}

export interface CapturedRegion {
  id: string
  itemId: string
  createdAt: number
  region: { x: number, y: number, width: number, height: number } // normalized 0-1
  // Cropped preview image for the panel (blob URL)
  imageUrl: string | null
  isCapturing: boolean
  // Region PPC
  ppcData: PpcResult | null
  ppcLoading: boolean
  paramsUsed: HsiParams
  displayUsed: PpcDisplayOptions
  // Overlays
  showOverlayOnMain: boolean
  showOverlayOnThumbnail: boolean
  overlayOpacity: number
  overlayDataUrl?: string | null // cached data URL for main viewer overlay (optional)
}

export interface RoiHueSample {
  x: number
  y: number
  width: number
  height: number
  output_width: number
  tissue_fraction?: number
  dab_band_fraction?: number
  dab_band_fraction_thumb?: number
  hue_value: number
  hue_width: number
}

export interface RoiHueSampleDebug {
  status: 'accepted' | 'rejected'
  reason?: 'tissue_fraction' | 'dab_band_fraction_thumb' | 'region_fetch_failed' | 'likely_negative_no_dab' | 'too_close' | null
  x: number
  y: number
  width: number
  height: number
  tissue_fraction?: number | null
  dab_band_fraction_thumb?: number | null
  dab_band_fraction?: number | null
  output_width?: number
  hue_value?: number
  hue_width?: number
}

export interface RoiHueSampleReport {
  item_id: string
  baseline: Record<string, any>
  rois: RoiHueSample[]
  debug_rois?: RoiHueSampleDebug[]
  summary: Record<string, any>
  metrics?: Record<string, any>
}

export interface SavedParameterSet {
  id: string
  name: string
  timestamp: number
  method: string
  // Source context (optional; added later for better traceability)
  itemId?: string
  itemName?: string
  savedFrom?: {
    viewport?: { x: number, y: number, width: number, height: number, zoom: number }
    region?: { x: number, y: number, width: number, height: number }
  }
  // Small reference preview image (optional; stored as a small data URL)
  preview?: {
    dataUrl: string
    width: number
    height: number
    overlayOpacity: number
    createdAt: number
  }
  // HSI parameters
  hueValue: number
  hueWidth: number
  saturationMinimum: number
  intensityUpperLimit: number
  intensityWeakThreshold: number
  intensityStrongThreshold: number
  intensityLowerLimit: number
  // Results
  results: PpcResult | null
  // Display options
  visible: boolean
  color?: string  // Optional custom color for overlay
}
