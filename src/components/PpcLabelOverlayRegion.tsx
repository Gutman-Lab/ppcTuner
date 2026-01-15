import { useState, useEffect } from 'react'

// PPC Label Overlay component for regions - shows classification results
export function PpcLabelOverlayRegion({ 
  itemId,
  x,
  y,
  width,
  height,
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
  x: number
  y: number
  width: number
  height: number
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
    // Build URL with all HSI parameters, display options, and region coordinates
    const params = new URLSearchParams({
      item_id: itemId,
      x: x.toString(),
      y: y.toString(),
      width: width.toString(),
      height: height.toString(),
      output_width: '1024',
      method: 'hsi',
      hue_value: (hueValue || 0.1).toString(),
      hue_width: (hueWidth || 0.1).toString(),
      saturation_minimum: (saturationMinimum || 0.1).toString(),
      intensity_upper_limit: (intensityUpperLimit || 0.9).toString(),
      intensity_weak_threshold: (intensityWeakThreshold || 0.6).toString(),
      intensity_strong_threshold: (intensityStrongThreshold || 0.3).toString(),
      intensity_lower_limit: (intensityLowerLimit || 0.05).toString(),
      show_weak: (showWeak ?? true).toString(),
      show_plain: (showPlain ?? true).toString(),
      show_strong: (showStrong ?? true).toString(),
      color_scheme: colorScheme || 'blue-green-red',
    })
    
    const labelUrl = `/api/ppc/label-image-region?${params.toString()}`
    
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
        console.error('Error loading region label image:', error)
        setLoading(false)
      })
    
    // Cleanup: revoke object URL when component unmounts or parameters change
    return () => {
      // Cleanup will be handled by the next effect run
    }
  }, [itemId, x, y, width, height, method, hueValue, hueWidth, saturationMinimum, intensityUpperLimit, intensityWeakThreshold, intensityStrongThreshold, intensityLowerLimit, showWeak, showPlain, showStrong, colorScheme])
  
  // Separate effect for cleanup when labelImage changes
  useEffect(() => {
    return () => {
      if (labelImage) {
        URL.revokeObjectURL(labelImage)
      }
    }
  }, [labelImage])
  
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
