import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'
import 'bdsa-react-components/styles.css'
import { ThumbnailGrid, FolderBrowser } from 'bdsa-react-components'
import type { Resource, Item } from 'bdsa-react-components'

// Simple histogram chart component
function HistogramChart({ data }: { data: any }) {
  const maxValue = Math.max(
    ...(data.histogram_r || []),
    ...(data.histogram_g || []),
    ...(data.histogram_b || [])
  )
  const bins = data.bins || 256
  const height = 200
  
  return (
    <div style={{ position: 'relative', height: `${height}px`, width: '100%' }}>
      <svg width="100%" height={height} style={{ border: '1px solid #ddd', borderRadius: '4px', backgroundColor: '#fff' }}>
        {/* Draw histograms */}
        {data.histogram_r && data.histogram_r.map((value: number, i: number) => {
          const barHeight = (value / maxValue) * (height - 30)
          const x = (i / bins) * 100
          const width = (100 / bins)
          return (
            <rect
              key={`r-${i}`}
              x={`${x}%`}
              y={height - 30 - barHeight}
              width={`${width}%`}
              height={barHeight}
              fill="rgba(255, 0, 0, 0.6)"
            />
          )
        })}
        {data.histogram_g && data.histogram_g.map((value: number, i: number) => {
          const barHeight = (value / maxValue) * (height - 30)
          const x = (i / bins) * 100
          const width = (100 / bins)
          return (
            <rect
              key={`g-${i}`}
              x={`${x}%`}
              y={height - 30 - barHeight}
              width={`${width}%`}
              height={barHeight}
              fill="rgba(0, 200, 0, 0.6)"
            />
          )
        })}
        {data.histogram_b && data.histogram_b.map((value: number, i: number) => {
          const barHeight = (value / maxValue) * (height - 30)
          const x = (i / bins) * 100
          const width = (100 / bins)
          return (
            <rect
              key={`b-${i}`}
              x={`${x}%`}
              y={height - 30 - barHeight}
              width={`${width}%`}
              height={barHeight}
              fill="rgba(0, 0, 255, 0.6)"
            />
          )
        })}
        {/* X-axis labels */}
        <text x="5%" y={height - 10} fontSize="11" fill="#333" fontWeight="500">0</text>
        <text x="50%" y={height - 10} fontSize="11" fill="#333" textAnchor="middle" fontWeight="500">128</text>
        <text x="95%" y={height - 10} fontSize="11" fill="#333" textAnchor="end" fontWeight="500">255</text>
        {/* Legend */}
        <g transform={`translate(10, 10)`}>
          <rect x="0" y="0" width="14" height="14" fill="rgba(255, 0, 0, 0.7)" stroke="#333" strokeWidth="0.5" />
          <text x="18" y="11" fontSize="12" fill="#333" fontWeight="500">Red</text>
          <rect x="55" y="0" width="14" height="14" fill="rgba(0, 200, 0, 0.7)" stroke="#333" strokeWidth="0.5" />
          <text x="73" y="11" fontSize="12" fill="#333" fontWeight="500">Green</text>
          <rect x="120" y="0" width="14" height="14" fill="rgba(0, 0, 255, 0.7)" stroke="#333" strokeWidth="0.5" />
          <text x="138" y="11" fontSize="12" fill="#333" fontWeight="500">Blue</text>
        </g>
      </svg>
    </div>
  )
}

// Tissue mask overlay component - now uses backend-generated mask
// PPC Label Overlay component - shows classification results
function PpcLabelOverlay({ 
  itemId, 
  method, 
  opacity,
  showWeak,
  showPlain,
  showStrong,
  colorScheme,
  // HSI parameters
  hueValue,
  hueWidth,
  saturationMinimum,
  intensityUpperLimit,
  intensityWeakThreshold,
  intensityStrongThreshold,
  intensityLowerLimit
}: { 
  itemId: string
  method: string
  opacity: number
  showWeak: boolean
  showPlain: boolean
  showStrong: boolean
  colorScheme: string
  hueValue?: number
  hueWidth?: number
  saturationMinimum?: number
  intensityUpperLimit?: number
  intensityWeakThreshold?: number
  intensityStrongThreshold?: number
  intensityLowerLimit?: number
}) {
  const [labelImage, setLabelImage] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  
  useEffect(() => {
    if (method !== 'hsi') {
      setLabelImage(null)
      setLoading(false)
      return
    }
    
    setLoading(true)
    // Build URL with all HSI parameters and display options
    const params = new URLSearchParams({
      item_id: itemId,
      method: 'hsi',
      thumbnail_width: '1024',
      hue_value: (hueValue || 0.1).toString(),
      hue_width: (hueWidth || 0.1).toString(),
      saturation_minimum: (saturationMinimum || 0.1).toString(),
      intensity_upper_limit: (intensityUpperLimit || 0.9).toString(),
      intensity_weak_threshold: (intensityWeakThreshold || 0.6).toString(),
      intensity_strong_threshold: (intensityStrongThreshold || 0.3).toString(),
      intensity_lower_limit: (intensityLowerLimit || 0.05).toString(),
      show_weak: showWeak.toString(),
      show_plain: showPlain.toString(),
      show_strong: showStrong.toString(),
      color_scheme: colorScheme,
    })
    
    const labelUrl = `/api/ppc/label-image?${params.toString()}`
    
    fetch(labelUrl)
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to load label image: ${response.status}`)
        }
        return response.blob()
      })
      .then(blob => {
        const url = URL.createObjectURL(blob)
        setLabelImage(url)
        setLoading(false)
      })
      .catch(error => {
        console.error('Error loading label image:', error)
        setLoading(false)
      })
    
    // Cleanup: revoke object URL when component unmounts or parameters change
    return () => {
      if (labelImage) {
        URL.revokeObjectURL(labelImage)
      }
    }
  }, [itemId, method, hueValue, hueWidth, saturationMinimum, intensityUpperLimit, intensityWeakThreshold, intensityStrongThreshold, intensityLowerLimit, showWeak, showPlain, showStrong, colorScheme])
  
  if (method !== 'hsi') return null
  
  if (loading) {
    return (
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0,0,0,0.1)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: '4px'
      }}>
        <span style={{ color: '#333', fontSize: '0.875rem' }}>Loading label overlay...</span>
      </div>
    )
  }
  
  if (!labelImage) return null
  
  return (
    <img
      src={labelImage}
      alt="PPC label overlay"
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        borderRadius: '4px',
        pointerEvents: 'none',
        opacity: opacity
      }}
    />
  )
}

function TissueMaskOverlay({ itemId, backgroundThreshold }: { itemId: string, backgroundThreshold: number }) {
  const [maskImage, setMaskImage] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  
  useEffect(() => {
    // Fetch mask from backend endpoint
    const maskUrl = `/api/images/${itemId}/mask?width=1024&background_threshold=${backgroundThreshold}`
    
    fetch(maskUrl)
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to load mask: ${response.status}`)
        }
        return response.blob()
      })
      .then(blob => {
        const url = URL.createObjectURL(blob)
        setMaskImage(url)
        setLoading(false)
      })
      .catch(error => {
        console.error('Error loading mask:', error)
        setLoading(false)
      })
    
    // Cleanup: revoke object URL when component unmounts or image changes
    return () => {
      if (maskImage) {
        URL.revokeObjectURL(maskImage)
      }
    }
  }, [itemId, backgroundThreshold])
  
  if (loading) {
    return (
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0,0,0,0.1)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: '4px'
      }}>
        <span style={{ color: '#333', fontSize: '0.875rem' }}>Loading mask...</span>
      </div>
    )
  }
  
  if (!maskImage) return null
  
  return (
    <img
      src={maskImage}
      alt="Tissue mask overlay"
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        borderRadius: '4px',
        pointerEvents: 'none',
        opacity: 0.7  // Make overlay semi-transparent
      }}
    />
  )
}

interface AppConfig {
  dsaBaseUrl: string
  startFolder: string
  startFolderType: string
  dsaToken: string | null  // Token obtained from API key authentication
}

interface PpcResult {
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

interface IntensityMap {
  item_id: string
  positive_intensities: number[]  // Intensity values (0-1) for positive pixels
  positive_pixel_indices: number[][]  // [row, col] pairs for positive pixels
  tissue_pixels: number
  total_pixels: number
  background_pixels: number
  image_shape: [number, number]  // [height, width]
  metrics: { [key: string]: number }
}

interface SavedParameterSet {
  id: string
  name: string
  timestamp: number
  method: string
  // HSI parameters
  hueValue: number
  hueWidth: number
  saturationMinimum: number
  intensityUpperLimit: number
  intensityWeakThreshold: number
  intensityStrongThreshold: number
  intensityLowerLimit: number
  // RGB parameters (if applicable)
  brownThreshold?: number
  yellowThreshold?: number
  redThreshold?: number
  // Results
  results: PpcResult | null
  // Display options
  visible: boolean
  color?: string  // Optional custom color for overlay
}

function App() {
  // Load saved state from localStorage
  const [selectedFolderId, setSelectedFolderId] = useState<string>(() => {
    const saved = localStorage.getItem('ppcTuner_selectedFolderId')
    return saved || ''
  })
  const [selectedItem, setSelectedItem] = useState<Item | null>(() => {
    const saved = localStorage.getItem('ppcTuner_selectedItem')
    if (saved) {
      try {
        return JSON.parse(saved)
      } catch (e) {
        return null
      }
    }
    return null
  })
  const [histogramData, setHistogramData] = useState<any>(null)
  const [histogramLoading, setHistogramLoading] = useState(false)
  const [showTissueMask, setShowTissueMask] = useState(false)
  const [ppcData, setPpcData] = useState<PpcResult | null>(null)
  const [ppcLoading, setPpcLoading] = useState(false)
  const [ppcMethod, setPpcMethod] = useState<string>('rgb_ratio')
  
  // Intensity map for real-time threshold adjustment (HSI method only)
  const [intensityMap, setIntensityMap] = useState<IntensityMap | null>(null)
  const [intensityMapLoading, setIntensityMapLoading] = useState<boolean>(false)
  // HSI parameters (declared early to avoid initialization order issues)
  const [hueValue, setHueValue] = useState<number>(0.1)
  const [hueWidth, setHueWidth] = useState<number>(0.1)
  const [saturationMinimum, setSaturationMinimum] = useState<number>(0.1)
  const [intensityUpperLimit, setIntensityUpperLimit] = useState<number>(0.9)
  const [intensityWeakThreshold, setIntensityWeakThreshold] = useState<number>(0.6)
  const [intensityStrongThreshold, setIntensityStrongThreshold] = useState<number>(0.3)
  const [intensityLowerLimit, setIntensityLowerLimit] = useState<number>(0.05)
  
  const [showPpcLabel, setShowPpcLabel] = useState<boolean>(false)
  const [ppcLabelOpacity, setPpcLabelOpacity] = useState<number>(0.5)
  const [showWeak, setShowWeak] = useState<boolean>(true)
  const [showPlain, setShowPlain] = useState<boolean>(true)
  const [showStrong, setShowStrong] = useState<boolean>(true)
  const [labelColorScheme, setLabelColorScheme] = useState<string>('blue-green-red')
  
  // Real-time adjustment mode
  const [autoCompute, setAutoCompute] = useState<boolean>(false)
  const computeDebounceTimerRef = useRef<NodeJS.Timeout | null>(null)
  const [savedParameterSets, setSavedParameterSets] = useState<SavedParameterSet[]>(() => {
    const saved = localStorage.getItem('ppcTuner_savedParameterSets')
    if (saved) {
      try {
        return JSON.parse(saved)
      } catch (e) {
        return []
      }
    }
    return []
  })
  const [nextSetId, setNextSetId] = useState<number>(() => {
    const saved = localStorage.getItem('ppcTuner_nextSetId')
    return saved ? parseInt(saved, 10) : 1
  })
  
  // Persist saved parameter sets to localStorage
  useEffect(() => {
    localStorage.setItem('ppcTuner_savedParameterSets', JSON.stringify(savedParameterSets))
  }, [savedParameterSets])
  
  useEffect(() => {
    localStorage.setItem('ppcTuner_nextSetId', nextSetId.toString())
  }, [nextSetId])

  // Classify intensities in real-time (client-side, no API call needed)
  const classifyIntensities = useCallback((
    intensities: number[],
    tissuePixels: number,
    totalPixels: number,
    backgroundPixels: number,
    itemId: string,
    weakThreshold: number,
    strongThreshold: number
  ): PpcResult => {
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
      }
    }
  }, [hueValue, hueWidth, saturationMinimum, intensityUpperLimit, intensityLowerLimit])
  
  // Helper function for rounding
  const round = (value: number, decimals: number) => {
    return Math.round(value * Math.pow(10, decimals)) / Math.pow(10, decimals)
  }

  // Fetch intensity map (when hue/saturation/bounds change, not thresholds)
  const fetchIntensityMap = useCallback(async (itemId: string) => {
    if (!itemId || ppcMethod !== 'hsi') return
    
    setIntensityMapLoading(true)
    try {
      const url = `/api/ppc/intensity-map?item_id=${itemId}&thumbnail_width=1024&hue_value=${hueValue}&hue_width=${hueWidth}&saturation_minimum=${saturationMinimum}&intensity_upper_limit=${intensityUpperLimit}&intensity_lower_limit=${intensityLowerLimit}`
      const response = await fetch(url)
      if (response.ok) {
        const data = await response.json() as IntensityMap
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
            intensityStrongThreshold
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
            intensityStrongThreshold
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
  }, [
    ppcMethod,
    hueValue,
    hueWidth,
    saturationMinimum,
    intensityUpperLimit,
    intensityLowerLimit,
    intensityWeakThreshold,
    intensityStrongThreshold,
    classifyIntensities
  ])

  // Reusable PPC computation function (memoized with useCallback)
  // For HSI method: uses intensity map if available and only thresholds changed
  // For RGB method: always uses full API call
  const computePpc = useCallback(async (itemId: string) => {
    if (!itemId) return
    
    // For HSI method: check if we can use intensity map (only thresholds changed)
    if (ppcMethod === 'hsi' && intensityMap && intensityMap.item_id === itemId) {
      // We have an intensity map - classify in real-time (instant!)
      const classified = classifyIntensities(
        intensityMap.positive_intensities,
        intensityMap.tissue_pixels,
        intensityMap.total_pixels,
        intensityMap.background_pixels,
        itemId,
        intensityWeakThreshold,
        intensityStrongThreshold
      )
      setPpcData(classified)
      return // No API call needed!
    }
    
    // Full API call needed (RGB method, or intensity map not available, or bounds changed)
    setPpcLoading(true)
    setPpcData(null)
    try {
      let url = `/api/ppc/compute?item_id=${itemId}&method=${ppcMethod}&thumbnail_width=1024`
      if (ppcMethod === 'hsi') {
        url += `&hue_value=${hueValue}&hue_width=${hueWidth}&saturation_minimum=${saturationMinimum}&intensity_upper_limit=${intensityUpperLimit}&intensity_weak_threshold=${intensityWeakThreshold}&intensity_strong_threshold=${intensityStrongThreshold}&intensity_lower_limit=${intensityLowerLimit}`
      }
      const response = await fetch(url)
      if (response.ok) {
        const data = await response.json()
        setPpcData(data)
      } else {
        const errorText = await response.text()
        console.error('PPC computation failed:', response.status, errorText)
        if (!autoCompute) { // Only show alert if manual compute
          alert(`PPC computation failed: ${response.status}`)
        }
      }
    } catch (error) {
      console.error('Error computing PPC:', error)
      if (!autoCompute) { // Only show alert if manual compute
        alert(`Error: ${error}`)
      }
    } finally {
      setPpcLoading(false)
    }
  }, [
    ppcMethod,
    hueValue,
    hueWidth,
    saturationMinimum,
    intensityUpperLimit,
    intensityWeakThreshold,
    intensityStrongThreshold,
    intensityLowerLimit,
    autoCompute,
    intensityMap,
    classifyIntensities
  ])

  // Fetch intensity map when hue/saturation/bounds change (not thresholds)
  // This allows instant threshold adjustment without API calls
  useEffect(() => {
    if (!autoCompute || !selectedItem || ppcMethod !== 'hsi') return
    
    const itemId = (selectedItem as any)._id || (selectedItem as any).id
    if (!itemId) return

    // Clear existing timer
    if (computeDebounceTimerRef.current) {
      clearTimeout(computeDebounceTimerRef.current)
    }

    // Set new timer (500ms debounce) - fetch intensity map when bounds change
    computeDebounceTimerRef.current = setTimeout(() => {
      fetchIntensityMap(itemId)
    }, 500)

    // Cleanup
    return () => {
      if (computeDebounceTimerRef.current) {
        clearTimeout(computeDebounceTimerRef.current)
      }
    }
  }, [
    autoCompute,
    selectedItem,
    ppcMethod,
    hueValue,
    hueWidth,
    saturationMinimum,
    intensityUpperLimit,
    intensityLowerLimit,
    fetchIntensityMap
  ])

  // Real-time threshold adjustment: when thresholds change, reclassify instantly
  useEffect(() => {
    if (!autoCompute || !intensityMap || ppcMethod !== 'hsi') return
    
    const itemId = (selectedItem as any)?._id || (selectedItem as any)?.id
    if (!itemId || intensityMap.item_id !== itemId) return
    
    // We have intensity map - classify in real-time (instant, no API call!)
    const classified = classifyIntensities(
      intensityMap.positive_intensities,
      intensityMap.tissue_pixels,
      intensityMap.total_pixels,
      intensityMap.background_pixels,
      itemId,
      intensityWeakThreshold,
      intensityStrongThreshold
    )
    setPpcData(classified)
  }, [
    autoCompute,
    intensityMap,
    selectedItem,
    ppcMethod,
    intensityWeakThreshold,
    intensityStrongThreshold,
    classifyIntensities
  ])
  
  // Save selectedFolderId to localStorage when it changes
  useEffect(() => {
    if (selectedFolderId) {
      localStorage.setItem('ppcTuner_selectedFolderId', selectedFolderId)
      console.log('📁 Selected folder ID saved:', selectedFolderId)
    } else {
      localStorage.removeItem('ppcTuner_selectedFolderId')
    }
  }, [selectedFolderId])
  
  // Save selectedItem to localStorage when it changes
  useEffect(() => {
    if (selectedItem) {
      localStorage.setItem('ppcTuner_selectedItem', JSON.stringify(selectedItem))
      console.log('🖼️ Selected item saved:', (selectedItem as any).name || (selectedItem as any)._id)
    } else {
      localStorage.removeItem('ppcTuner_selectedItem')
    }
  }, [selectedItem])
  
  const [apiBaseUrl, setApiBaseUrl] = useState<string>('http://bdsa.pathology.emory.edu:8080/api/v1')
  const [dsaToken, setDsaToken] = useState<string | null>(null)
  
  // Memoize fetchFn for ThumbnailGrid to prevent unnecessary re-fetches
  const thumbnailGridFetchFn = useCallback(async (url: string, options?: RequestInit) => {
    if (!dsaToken) return fetch(url, options)
    console.log('ThumbnailGrid fetch:', url)
    console.log('ThumbnailGrid folderId:', selectedFolderId)
    const response = await fetch(url, {
      ...options,
      headers: {
        ...options?.headers,
        'Girder-Token': dsaToken,
      },
    })
    console.log('ThumbnailGrid response status:', response.status, 'for', url)
    
    // Clone the response for debugging without consuming the original
    if (response.ok) {
      const clonedResponse = response.clone()
      try {
        const data = await clonedResponse.json()
        console.log('ThumbnailGrid response data:', data)
        console.log('Number of items:', Array.isArray(data) ? data.length : 'Not an array')
      } catch (e) {
        console.log('Could not parse response for logging:', e)
      }
    } else {
      const clonedResponse = response.clone()
      try {
        const text = await clonedResponse.text()
        console.error('ThumbnailGrid error response:', response.status, text)
      } catch (e) {
        console.error('ThumbnailGrid error status:', response.status)
      }
    }
    
    // Return the original, unconsumed response
    return response
  }, [dsaToken, selectedFolderId])
  
  // Memoize fetchFn for FolderBrowser to prevent unnecessary re-fetches
  const folderBrowserFetchFn = useCallback(async (url: string, options?: RequestInit) => {
    if (!dsaToken) return fetch(url, options)
    console.log('FolderBrowser fetch:', url)
    const headers = {
      ...options?.headers,
      'Girder-Token': dsaToken,
    }
    const response = await fetch(url, {
      ...options,
      headers,
    })
    console.log('FolderBrowser response status:', response.status, 'for', url)
    if (!response.ok) {
      const errorText = await response.text()
      console.error('FolderBrowser error:', response.status, errorText)
    }
    return response
  }, [dsaToken])
  
  const [configLoaded, setConfigLoaded] = useState(false)
  const [startFolder, setStartFolder] = useState<string>('')
  const [startFolderType, setStartFolderType] = useState<string>('collection')

  // Load configuration from backend
  useEffect(() => {
    async function loadConfig() {
      try {
        const response = await fetch('/api/config')
        if (response.ok) {
          const config: AppConfig = await response.json()
          console.log('Config loaded:', {
            dsaBaseUrl: config.dsaBaseUrl,
            hasToken: !!config.dsaToken,
            tokenLength: config.dsaToken?.length || 0,
            tokenPreview: config.dsaToken ? config.dsaToken.substring(0, 20) + '...' : 'MISSING',
          })
          setApiBaseUrl(config.dsaBaseUrl)
          if (config.dsaToken) {
            console.log('DSA token loaded from backend, length:', config.dsaToken.length)
            console.log('Token first 20 chars:', config.dsaToken.substring(0, 20), '...')
            setDsaToken(config.dsaToken)
            
            // Test the token by checking if it's valid
            const testUserUrl = `${config.dsaBaseUrl}/user/me`
            console.log('Testing token validity with:', testUserUrl)
            fetch(testUserUrl, {
              headers: { 
                'Girder-Token': config.dsaToken || '',
              },
            })
              .then(async (res) => {
                if (res.ok) {
                  const userData = await res.json()
                  if (userData) {
                    console.log('✓ Token is valid. Authenticated as:', userData.login || userData.name || userData._id || 'user')
                  } else {
                    console.log('✓ Token is valid (user data is null)')
                  }
                  
                  // Test access to the start collection
                  if (config.startFolder && config.startFolderType === 'collection') {
                    const testUrl = `${config.dsaBaseUrl}/collection/${config.startFolder}`
                    console.log('Testing access to start collection:', testUrl)
                    return fetch(testUrl, {
                      headers: { 'Girder-Token': config.dsaToken || '' },
                    })
                      .then(async (collRes) => {
                        if (collRes.ok) {
                          const data = await collRes.json()
                          console.log('✓ Can access start collection:', data.name || data._id)
                        } else {
                          const errorText = await collRes.text()
                          console.error('✗ Cannot access start collection:', collRes.status, collRes.statusText)
                          console.error('Error response:', errorText)
                        }
                      })
                  }
                } else {
                  const errorText = await res.text()
                  console.error('✗ Token is INVALID:', res.status, res.statusText)
                  console.error('Error response:', errorText)
                }
              })
              .catch((err) => console.error('Token test error:', err))
          } else {
            console.warn('No DSA token found - API key may not be set or authentication failed')
          }
          // Store start folder info
          if (config.startFolder) {
            setStartFolder(config.startFolder)
            setStartFolderType(config.startFolderType)
            // If start folder is a folder (not collection) and no saved folder, set it as selected
            if (config.startFolderType === 'folder' && !selectedFolderId) {
              setSelectedFolderId(config.startFolder)
            }
          }
          
          // If we have a saved selected item, load its histogram
          if (selectedItem && config.dsaToken) {
            const itemId = (selectedItem as any)._id || (selectedItem as any).id
            if (itemId) {
              console.log('🔄 Restoring saved item and loading histogram:', itemId)
              setHistogramLoading(true)
              try {
                const histResponse = await fetch(`/api/ppc/histogram?item_id=${itemId}&width=1024&bins=256`)
                if (histResponse.ok) {
                  const histData = await histResponse.json()
                  setHistogramData(histData)
                  console.log('✓ Histogram loaded for saved item')
                }
              } catch (error) {
                console.error('Failed to load histogram for saved item:', error)
              } finally {
                setHistogramLoading(false)
              }
            }
          }
        }
      } catch (error) {
        console.error('Failed to load config:', error)
      } finally {
        setConfigLoaded(true)
      }
    }
    loadConfig()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // Only run on mount - selectedItem and selectedFolderId are intentionally excluded

  const handleResourceSelect = (resource: Resource) => {
    console.log('Resource selected:', resource)
    
    // Resource object uses _id (MongoDB/Girder convention), not id
    const resourceId = (resource as any)._id || (resource as any).id
    const resourceType = (resource as any)._modelType || resource.type
    
    console.log('Resource type:', resourceType)
    console.log('Resource ID:', resourceId, 'Type:', typeof resourceId)
    
    // Handle both folders and collections - collections can contain items too
    if ((resourceType === 'folder' || resource.type === 'folder') && typeof resourceId === 'string') {
      console.log('✓ Setting selected folder ID:', resourceId)
      setSelectedFolderId(resourceId)
    } else if ((resourceType === 'collection' || resource.type === 'collection') && typeof resourceId === 'string') {
      // Collections can also contain items, so allow selecting them
      console.log('✓ Setting selected collection ID:', resourceId)
      setSelectedFolderId(resourceId)
    } else {
      console.log('⚠ Resource is not a folder/collection or missing ID:', resource)
      setSelectedFolderId('')
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>PPC Tuner</h1>
        <p>Load images from DSA and generate aBeta-stained slide thumbnails</p>
        {dsaToken && <p style={{ fontSize: '0.85rem', opacity: 0.8 }}>✓ Authenticated with DSA (token obtained from API key)</p>}
        {!dsaToken && configLoaded && <p style={{ fontSize: '0.85rem', color: '#e74c3c' }}>⚠ No authentication token - check DSAKEY in .env and backend logs</p>}
      </header>

      <main className="app-main">
        <div className="browser-section">
          <h2 style={{ margin: 0, marginBottom: '0.5rem' }}>Select Folder</h2>
          {configLoaded && (
            <FolderBrowser
              apiBaseUrl={apiBaseUrl}
              showCollections={true}
              onResourceSelect={handleResourceSelect}
              foldersPerPage={50}
              // Start at a specific folder/collection using rootId and rootType
              // Based on Storybook controls: rootId and rootType props
              {...(startFolder 
                ? { 
                    rootId: startFolder,
                    rootType: startFolderType as 'collection' | 'folder'
                  } 
                : {})}
              apiHeaders={dsaToken ? { 'Girder-Token': dsaToken } : undefined}
              fetchFn={folderBrowserFetchFn}
            />
          )}
        </div>

        {selectedFolderId ? (
          <div className="thumbnail-section">
            <h2>Thumbnails</h2>
            <ThumbnailGrid
              key={selectedFolderId} // Force re-render when folder changes
              apiBaseUrl={apiBaseUrl}
              folderId={selectedFolderId}
              thumbnailSize="l"
              itemsPerPage={12}
              apiHeaders={dsaToken ? { 'Girder-Token': dsaToken } : undefined}
              tokenQueryParam={!!dsaToken}
              fetchFn={thumbnailGridFetchFn}
              onThumbnailClick={async (item) => {
                console.log('Thumbnail clicked:', item)
                setSelectedItem(item)
                setHistogramData(null)
                setPpcData(null) // Clear PPC data when new image is selected
                setIntensityMap(null) // Clear intensity map when new image is selected
                setShowPpcLabel(false) // Hide label overlay when new image is selected
                setHistogramLoading(true)
                
                // Fetch histogram data
                const itemId = (item as any)._id || (item as any).id
                if (itemId) {
                  try {
                    const response = await fetch(`/api/ppc/histogram?item_id=${itemId}&width=1024&bins=256`)
                    if (response.ok) {
                      const data = await response.json()
                      setHistogramData(data)
                    } else {
                      console.error('Failed to fetch histogram:', response.status)
                    }
                  } catch (error) {
                    console.error('Error fetching histogram:', error)
                  } finally {
                    setHistogramLoading(false)
                  }
                }
              }}
            />
          </div>
        ) : (
          <div className="thumbnail-section" style={{ opacity: 0.5 }}>
            <h2>Thumbnails</h2>
            <p>Select a folder to view thumbnails</p>
          </div>
        )}

        {selectedItem && (() => {
          const itemId = (selectedItem as any)._id || (selectedItem as any).id
          if (!itemId) return null
          
          // Use backend proxy endpoint to avoid CORS issues
          const thumbnailUrl = `/api/images/${itemId}/thumbnail?width=1024`
          
          // Reference to TissueMaskOverlay to avoid scope issues in IIFE
          const MaskOverlay = TissueMaskOverlay
          
          return (
            <div className="image-viewer-section" style={{ marginTop: '2rem' }}>
              <h2>Selected Image</h2>
              <div style={{ 
                display: 'flex',
                gap: '1.5rem',
                alignItems: 'flex-start',
                maxWidth: '1600px',
                margin: '0 auto'
              }}>
                {/* Parameter Sets Panel - Left */}
                <div style={{ 
                  flex: '0 0 280px',
                  border: '1px solid #ddd', 
                  borderRadius: '8px', 
                  padding: '1rem',
                  backgroundColor: '#fff',
                  maxHeight: '80vh',
                  overflowY: 'auto'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <h3 style={{ margin: 0, fontSize: '1rem', color: '#333' }}>Parameter Sets</h3>
                    <button
                      onClick={() => {
                        if (!ppcData) {
                          alert('Please compute PPC first before saving parameters')
                          return
                        }
                        const newSet: SavedParameterSet = {
                          id: `set-${nextSetId}`,
                          name: `Run ${nextSetId}`,
                          timestamp: Date.now(),
                          method: ppcMethod,
                          hueValue,
                          hueWidth,
                          saturationMinimum,
                          intensityUpperLimit,
                          intensityWeakThreshold,
                          intensityStrongThreshold,
                          intensityLowerLimit,
                          brownThreshold: ppcMethod === 'rgb_ratio' ? undefined : undefined, // Will add RGB params if needed
                          yellowThreshold: ppcMethod === 'rgb_ratio' ? undefined : undefined,
                          redThreshold: ppcMethod === 'rgb_ratio' ? undefined : undefined,
                          results: ppcData,
                          visible: true,
                        }
                        setSavedParameterSets([...savedParameterSets, newSet])
                        setNextSetId(nextSetId + 1)
                      }}
                      disabled={!ppcData}
                      style={{
                        padding: '0.4rem 0.8rem',
                        fontSize: '0.75rem',
                        backgroundColor: ppcData ? '#4CAF50' : '#ccc',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: ppcData ? 'pointer' : 'not-allowed',
                        opacity: ppcData ? 1 : 0.6
                      }}
                    >
                      Save Current
                    </button>
                  </div>
                  
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
                                  // Load parameters into current controls
                                  setPpcMethod(set.method)
                                  setHueValue(set.hueValue)
                                  setHueWidth(set.hueWidth)
                                  setSaturationMinimum(set.saturationMinimum)
                                  setIntensityUpperLimit(set.intensityUpperLimit)
                                  setIntensityWeakThreshold(set.intensityWeakThreshold)
                                  setIntensityStrongThreshold(set.intensityStrongThreshold)
                                  setIntensityLowerLimit(set.intensityLowerLimit)
                                  // Optionally auto-compute with these parameters
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
                                  setSavedParameterSets(savedParameterSets.filter(s => s.id !== set.id))
                                }}
                                style={{
                                  padding: '0.2rem 0.4rem',
                                  fontSize: '0.7rem',
                                  backgroundColor: '#f44336',
                                  color: 'white',
                                  border: 'none',
                                  borderRadius: '3px',
                                  cursor: 'pointer',
                                  marginLeft: '0.25rem'
                                }}
                                title="Delete this parameter set"
                              >
                                ×
                              </button>
                            </div>
                          </div>
                          
                          <div style={{ fontSize: '0.75rem', color: '#666', marginBottom: '0.5rem' }}>
                            {new Date(set.timestamp).toLocaleTimeString()}
                          </div>
                          
                          {set.results && (
                            <div style={{ fontSize: '0.75rem', color: '#333', lineHeight: '1.6' }}>
                              {set.method === 'hsi' ? (
                                <>
                                  <div><strong>Weak:</strong> {set.results.weak_positive_pixels?.toLocaleString()} ({set.results.weak_percentage?.toFixed(1)}%)</div>
                                  <div><strong>Plain:</strong> {set.results.plain_positive_pixels?.toLocaleString()} ({set.results.plain_percentage?.toFixed(1)}%)</div>
                                  <div><strong>Strong:</strong> {set.results.strong_positive_pixels?.toLocaleString()} ({set.results.strong_percentage?.toFixed(1)}%)</div>
                                  <div style={{ marginTop: '0.25rem', fontWeight: 'bold' }}>
                                    <strong>Total:</strong> {set.results.total_positive_pixels?.toLocaleString()} ({set.results.positive_percentage?.toFixed(1)}%)
                                  </div>
                                </>
                              ) : (
                                <>
                                  <div><strong>Brown:</strong> {set.results.brown_pixels?.toLocaleString()} ({set.results.brown_percentage?.toFixed(1)}%)</div>
                                  <div><strong>Yellow:</strong> {set.results.yellow_pixels?.toLocaleString()} ({set.results.yellow_percentage?.toFixed(1)}%)</div>
                                  <div><strong>Red:</strong> {set.results.red_pixels?.toLocaleString()} ({set.results.red_percentage?.toFixed(1)}%)</div>
                                  <div style={{ marginTop: '0.25rem', fontWeight: 'bold' }}>
                                    <strong>Total:</strong> {set.results.total_positive_pixels?.toLocaleString()} ({set.results.positive_percentage?.toFixed(1)}%)
                                  </div>
                                </>
                              )}
                            </div>
                          )}
                          
                          {set.method === 'hsi' && (
                            <div style={{ fontSize: '0.7rem', color: '#999', marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid #eee' }}>
                              H: {set.hueValue.toFixed(2)}, W: {set.hueWidth.toFixed(2)}<br />
                              Weak≥{set.intensityWeakThreshold.toFixed(2)}, Strong&lt;{set.intensityStrongThreshold.toFixed(2)}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                
                {/* Image in the middle */}
                <div style={{ 
                  flex: '1 1 60%',
                  border: '1px solid #ddd', 
                  borderRadius: '8px', 
                  padding: '1rem',
                  backgroundColor: '#fff',
                  textAlign: 'center'
                }}>
                  <div style={{ marginBottom: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', alignItems: 'center' }}>
                    <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                      <label style={{ fontSize: '0.875rem', color: '#333', display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                        <input
                          type="checkbox"
                          checked={showTissueMask}
                          onChange={(e) => setShowTissueMask(e.target.checked)}
                          style={{ cursor: 'pointer' }}
                        />
                        Show Tissue Mask
                      </label>
                      {ppcData?.method === 'hsi' && (
                        <label style={{ fontSize: '0.875rem', color: '#333', display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                          <input
                            type="checkbox"
                            checked={showPpcLabel}
                            onChange={(e) => setShowPpcLabel(e.target.checked)}
                            style={{ cursor: 'pointer' }}
                          />
                          Show PPC Labels
                        </label>
                      )}
                    </div>
                    {showPpcLabel && ppcData?.method === 'hsi' && (
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
                    )}
                  </div>
                  <div style={{ position: 'relative', display: 'inline-block' }}>
                    <img 
                      src={thumbnailUrl}
                      alt={selectedItem.name || 'Selected image'}
                      style={{
                        maxWidth: '100%',
                        height: 'auto',
                        borderRadius: '4px',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                        display: 'block'
                      }}
                      onError={(e) => {
                        console.error('Failed to load thumbnail:', thumbnailUrl)
                        e.currentTarget.style.display = 'none'
                      }}
                    />
                    {showTissueMask && histogramData?.tissue_analysis && (
                      <MaskOverlay 
                        itemId={itemId}
                        backgroundThreshold={histogramData.tissue_analysis.background_threshold || 240}
                      />
                    )}
                    {/* Current PPC Label Overlay */}
                    {showPpcLabel && ppcData?.method === 'hsi' && (
                      <PpcLabelOverlay
                        itemId={itemId}
                        method={ppcMethod}
                        opacity={ppcLabelOpacity}
                        showWeak={showWeak}
                        showPlain={showPlain}
                        showStrong={showStrong}
                        colorScheme={labelColorScheme}
                        hueValue={hueValue}
                        hueWidth={hueWidth}
                        saturationMinimum={saturationMinimum}
                        intensityUpperLimit={intensityUpperLimit}
                        intensityWeakThreshold={intensityWeakThreshold}
                        intensityStrongThreshold={intensityStrongThreshold}
                        intensityLowerLimit={intensityLowerLimit}
                      />
                    )}
                    
                    {/* Saved Parameter Set Overlays */}
                    {savedParameterSets
                      .filter(set => set.visible && set.method === 'hsi')
                      .map((set, index) => (
                        <PpcLabelOverlay
                          key={set.id}
                          itemId={itemId}
                          method={set.method}
                          opacity={ppcLabelOpacity * 0.7} // Slightly more transparent for comparison
                          showWeak={showWeak}
                          showPlain={showPlain}
                          showStrong={showStrong}
                          colorScheme={labelColorScheme}
                          hueValue={set.hueValue}
                          hueWidth={set.hueWidth}
                          saturationMinimum={set.saturationMinimum}
                          intensityUpperLimit={set.intensityUpperLimit}
                          intensityWeakThreshold={set.intensityWeakThreshold}
                          intensityStrongThreshold={set.intensityStrongThreshold}
                          intensityLowerLimit={set.intensityLowerLimit}
                        />
                      ))}
                  </div>
                  <div style={{ marginTop: '1rem', fontSize: '0.875rem', color: '#333' }}>
                    <p><strong>Name:</strong> {selectedItem.name || 'N/A'}</p>
                    {itemId && <p><strong>ID:</strong> {itemId}</p>}
                    {histogramData?.tissue_analysis && (
                      <p style={{ marginTop: '0.5rem' }}>
                        <strong>Tissue:</strong> {histogramData.tissue_analysis.tissue_percentage.toFixed(1)}% | 
                        <strong> Background:</strong> {(100 - histogramData.tissue_analysis.tissue_percentage).toFixed(1)}%
                      </p>
                    )}
                  </div>
                </div>
                
                {/* Histogram and PPC on the right */}
                <div style={{ 
                  flex: '1 1 40%',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '1rem'
                }}>
                  {/* Histogram */}
                  <div style={{ 
                    border: '1px solid #ddd', 
                    borderRadius: '8px', 
                    padding: '1rem',
                    backgroundColor: '#fff',
                    minWidth: '300px'
                  }}>
                    <h3 style={{ marginTop: 0, marginBottom: '1rem', color: '#333' }}>Color Histogram</h3>
                    {histogramLoading ? (
                      <p style={{ color: '#666' }}>Loading histogram...</p>
                    ) : histogramData ? (
                      <div>
                        <HistogramChart data={histogramData} />
                        {histogramData.statistics && (
                          <div style={{ marginTop: '1rem', fontSize: '0.875rem', color: '#333', backgroundColor: '#f5f5f5', padding: '0.75rem', borderRadius: '4px' }}>
                            <p style={{ margin: '0 0 0.5rem 0', fontWeight: 'bold', color: '#333' }}>Statistics:</p>
                            <p style={{ margin: '0.25rem 0', color: '#333' }}>
                              <strong>Mean</strong> - R: {histogramData.statistics.mean.r.toFixed(1)}, G: {histogramData.statistics.mean.g.toFixed(1)}, B: {histogramData.statistics.mean.b.toFixed(1)}
                            </p>
                            <p style={{ margin: '0.25rem 0', color: '#333' }}>
                              <strong>Std</strong> - R: {histogramData.statistics.std.r.toFixed(1)}, G: {histogramData.statistics.std.g.toFixed(1)}, B: {histogramData.statistics.std.b.toFixed(1)}
                            </p>
                          </div>
                        )}
                      </div>
                    ) : (
                      <p style={{ color: '#666' }}>Click an image to load histogram</p>
                    )}
                  </div>

                  {/* PPC Section */}
                  <div style={{ 
                    border: '1px solid #ddd', 
                    borderRadius: '8px', 
                    padding: '1rem',
                    backgroundColor: '#fff',
                    minWidth: '300px'
                  }}>
                    <h3 style={{ marginTop: 0, marginBottom: '1rem', color: '#333' }}>Positive Pixel Count</h3>
                    <div style={{ marginBottom: '1rem', display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
                      <label style={{ fontSize: '0.875rem', color: '#333', marginRight: '0.5rem' }}>Method:</label>
                      <select
                        value={ppcMethod}
                        onChange={(e) => setPpcMethod(e.target.value)}
                        style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem', borderRadius: '4px', border: '1px solid #ddd' }}
                      >
                        <option value="rgb_ratio">RGB Ratio</option>
                        <option value="hsi">HSI (HistomicsTK)</option>
                      </select>
                    </div>

                    {/* HSI Parameters (only show when HSI method is selected) */}
                    {ppcMethod === 'hsi' && (
                      <div style={{ marginBottom: '1rem', padding: '0.75rem', backgroundColor: '#f9f9f9', borderRadius: '4px', fontSize: '0.875rem' }}>
                        <div style={{ marginBottom: '0.75rem' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                            <label style={{ color: '#333', fontWeight: '500' }}>Hue Value:</label>
                            <span style={{ color: '#666', fontSize: '0.8rem' }}>{(hueValue * 360).toFixed(1)}° ({hueValue.toFixed(3)})</span>
                          </div>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.001"
                            value={hueValue}
                            onChange={(e) => setHueValue(parseFloat(e.target.value))}
                            style={{ width: '100%' }}
                          />
                        </div>
                        <div style={{ marginBottom: '0.75rem' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                            <label style={{ color: '#333', fontWeight: '500' }}>Hue Width:</label>
                            <span style={{ color: '#666', fontSize: '0.8rem' }}>{(hueWidth * 360).toFixed(1)}° ({hueWidth.toFixed(3)})</span>
                          </div>
                          <input
                            type="range"
                            min="0.01"
                            max="0.5"
                            step="0.01"
                            value={hueWidth}
                            onChange={(e) => setHueWidth(parseFloat(e.target.value))}
                            style={{ width: '100%' }}
                          />
                        </div>
                        
                        {/* Intensity Threshold Controls */}
                        <div style={{ marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid #ddd' }}>
                          <div style={{ marginBottom: '0.5rem', fontWeight: 'bold', color: '#333' }}>Intensity Thresholds:</div>
                          <div style={{ marginBottom: '0.75rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                              <label style={{ color: '#333', fontWeight: '500' }}>Weak Threshold (≥):</label>
                              <span style={{ color: '#666', fontSize: '0.8rem' }}>{intensityWeakThreshold.toFixed(3)}</span>
                            </div>
                            <input
                              type="range"
                              min={Math.max(0.1, intensityStrongThreshold + 0.05)}
                              max="1"
                              step="0.01"
                              value={intensityWeakThreshold}
                              onChange={(e) => {
                                const newValue = parseFloat(e.target.value)
                                setIntensityWeakThreshold(newValue)
                                // Ensure strong threshold is always less than weak threshold
                                if (newValue <= intensityStrongThreshold) {
                                  setIntensityStrongThreshold(Math.max(0.05, newValue - 0.05))
                                }
                              }}
                              style={{ width: '100%' }}
                            />
                            <div style={{ fontSize: '0.75rem', color: '#999', marginTop: '0.25rem' }}>
                              Weak = intensity ≥ {intensityWeakThreshold.toFixed(2)} (lighter staining)
                            </div>
                          </div>
                          <div style={{ marginBottom: '0.75rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                              <label style={{ color: '#333', fontWeight: '500' }}>Strong Threshold (&lt;):</label>
                              <span style={{ color: '#666', fontSize: '0.8rem' }}>{intensityStrongThreshold.toFixed(3)}</span>
                            </div>
                            <input
                              type="range"
                              min="0.05"
                              max={Math.min(0.95, intensityWeakThreshold - 0.05)}
                              step="0.01"
                              value={intensityStrongThreshold}
                              onChange={(e) => {
                                const newValue = parseFloat(e.target.value)
                                setIntensityStrongThreshold(newValue)
                                // Ensure weak threshold is always greater than strong threshold
                                if (newValue >= intensityWeakThreshold) {
                                  setIntensityWeakThreshold(Math.min(1.0, newValue + 0.05))
                                }
                              }}
                              style={{ width: '100%' }}
                            />
                            <div style={{ fontSize: '0.75rem', color: '#999', marginTop: '0.25rem' }}>
                              Strong = intensity &lt; {intensityStrongThreshold.toFixed(2)} (darker staining)
                            </div>
                          </div>
                          <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.5rem', padding: '0.5rem', backgroundColor: '#fff', borderRadius: '4px' }}>
                            <strong>Plain</strong> = {intensityStrongThreshold.toFixed(2)} ≤ intensity &lt; {intensityWeakThreshold.toFixed(2)} (medium staining)
                          </div>
                        </div>
                        {/* Visual Hue Range Indicator */}
                        <div style={{ marginBottom: '0.75rem', padding: '0.5rem', backgroundColor: '#fff', borderRadius: '4px', border: '1px solid #ddd' }}>
                          <div style={{ fontSize: '0.75rem', color: '#666', marginBottom: '0.25rem' }}>Hue Range:</div>
                          <div style={{ 
                            height: '30px', 
                            borderRadius: '4px',
                            background: `linear-gradient(to right, 
                              hsl(${(hueValue - hueWidth/2) * 360}, 70%, 50%),
                              hsl(${hueValue * 360}, 70%, 50%),
                              hsl(${(hueValue + hueWidth/2) * 360}, 70%, 50%))`,
                            border: '1px solid #ccc',
                            position: 'relative'
                          }}>
                            <div style={{
                              position: 'absolute',
                              left: '50%',
                              top: '50%',
                              transform: 'translate(-50%, -50%)',
                              width: '2px',
                              height: '100%',
                              backgroundColor: '#000',
                              opacity: 0.5
                            }} />
                          </div>
                          <div style={{ fontSize: '0.7rem', color: '#999', marginTop: '0.25rem', textAlign: 'center' }}>
                            {(hueValue - hueWidth/2) * 360}° - {(hueValue + hueWidth/2) * 360}°
                          </div>
                        </div>
                        <button
                          onClick={async () => {
                            if (!itemId) return
                            try {
                              const response = await fetch(`/api/ppc/auto-detect-hue?item_id=${itemId}&thumbnail_width=1024`)
                              if (response.ok) {
                                const data = await response.json()
                                setHueValue(data.hue_value)
                                setHueWidth(data.hue_width)
                                setSaturationMinimum(data.saturation_minimum)
                                setIntensityUpperLimit(data.intensity_upper_limit)
                                setIntensityWeakThreshold(data.intensity_weak_threshold)
                                setIntensityStrongThreshold(data.intensity_strong_threshold)
                                setIntensityLowerLimit(data.intensity_lower_limit)
                              } else {
                                console.error('Auto-detect failed:', response.status)
                              }
                            } catch (error) {
                              console.error('Error auto-detecting hue:', error)
                            }
                          }}
                          disabled={!itemId}
                          style={{
                            padding: '0.4rem 0.8rem',
                            fontSize: '0.8rem',
                            backgroundColor: '#2196F3',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: !itemId ? 'not-allowed' : 'pointer',
                            opacity: !itemId ? 0.6 : 1,
                            width: '100%'
                          }}
                        >
                          Auto-detect Hue Parameters
                        </button>
                      </div>
                    )}

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
                    <button
                      onClick={() => {
                        const itemId = (selectedItem as any)?._id || (selectedItem as any)?.id
                        if (itemId) computePpc(itemId)
                      }}
                      disabled={ppcLoading || !selectedItem}
                      style={{
                        padding: '0.5rem 1rem',
                        fontSize: '0.875rem',
                        backgroundColor: autoCompute ? '#2196F3' : '#4CAF50',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: ppcLoading || !selectedItem ? 'not-allowed' : 'pointer',
                        opacity: ppcLoading || !selectedItem ? 0.6 : 1,
                        width: '100%',
                        marginBottom: '1rem'
                      }}
                    >
                      {ppcLoading || intensityMapLoading ? 'Computing...' : autoCompute ? 'Force Compute Now' : 'Compute PPC'}
                    </button>
                    {(ppcLoading || intensityMapLoading) ? (
                      <p style={{ color: '#666' }}>
                        {intensityMapLoading ? 'Fetching intensity map...' : 'Computing PPC...'}
                      </p>
                    ) : ppcData ? (
                      <div style={{ fontSize: '0.875rem', color: '#333', backgroundColor: '#f5f5f5', padding: '0.75rem', borderRadius: '4px' }}>
                        <p style={{ margin: '0 0 0.5rem 0', fontWeight: 'bold', color: '#333' }}>Results ({ppcData.method}):</p>
                        {ppcData.method === 'rgb_ratio' ? (
                          <>
                            <p style={{ margin: '0.25rem 0', color: '#333' }}>
                              <strong>Brown:</strong> {ppcData.brown_pixels?.toLocaleString()} pixels ({ppcData.brown_percentage?.toFixed(2)}%)
                            </p>
                            <p style={{ margin: '0.25rem 0', color: '#333' }}>
                              <strong>Yellow:</strong> {ppcData.yellow_pixels?.toLocaleString()} pixels ({ppcData.yellow_percentage?.toFixed(2)}%)
                            </p>
                            <p style={{ margin: '0.25rem 0', color: '#333' }}>
                              <strong>Red:</strong> {ppcData.red_pixels?.toLocaleString()} pixels ({ppcData.red_percentage?.toFixed(2)}%)
                            </p>
                            <p style={{ margin: '0.5rem 0 0.25rem 0', fontWeight: 'bold', color: '#333' }}>
                              <strong>Total Positive:</strong> {ppcData.total_positive_pixels?.toLocaleString()} pixels ({ppcData.positive_percentage?.toFixed(2)}%)
                            </p>
                          </>
                        ) : (
                          <>
                            <p style={{ margin: '0.25rem 0', color: '#333' }}>
                              <strong>Weak:</strong> {ppcData.weak_positive_pixels?.toLocaleString()} pixels ({ppcData.weak_percentage?.toFixed(2)}%)
                            </p>
                            <p style={{ margin: '0.25rem 0', color: '#333' }}>
                              <strong>Plain:</strong> {ppcData.plain_positive_pixels?.toLocaleString()} pixels ({ppcData.plain_percentage?.toFixed(2)}%)
                            </p>
                            <p style={{ margin: '0.25rem 0', color: '#333' }}>
                              <strong>Strong:</strong> {ppcData.strong_positive_pixels?.toLocaleString()} pixels ({ppcData.strong_percentage?.toFixed(2)}%)
                            </p>
                            <p style={{ margin: '0.5rem 0 0.25rem 0', fontWeight: 'bold', color: '#333' }}>
                              <strong>Total Positive:</strong> {ppcData.total_positive_pixels?.toLocaleString()} pixels ({ppcData.positive_percentage?.toFixed(2)}%)
                            </p>
                          </>
                        )}
                        <p style={{ margin: '0.5rem 0 0.25rem 0', color: '#666', fontSize: '0.75rem' }}>
                          Tissue: {ppcData.tissue_pixels?.toLocaleString()} | Background: {ppcData.background_pixels?.toLocaleString()}
                        </p>
                        {ppcData.metrics && (
                          <p style={{ margin: '0.5rem 0 0 0', color: '#666', fontSize: '0.75rem' }}>
                            {ppcData.metrics.execution_time_seconds === 0.0 ? (
                              <span style={{ color: '#4CAF50', fontWeight: 'bold' }}>⚡ Instant classification (client-side, no API call)</span>
                            ) : (
                              <>
                                Time: {ppcData.metrics.execution_time_seconds}s | CPU: {ppcData.metrics.cpu_percent}% | Memory: {ppcData.metrics.memory_mb?.toFixed(1)}MB
                              </>
                            )}
                          </p>
                        )}
                      </div>
                    ) : (
                      <p style={{ color: '#666' }}>Click "Compute PPC" to analyze positive pixels</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )
        })()}
      </main>
    </div>
  )
}

export default App
