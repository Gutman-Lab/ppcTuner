import { useCallback, useEffect } from 'react'
import type { Dispatch, SetStateAction } from 'react'
import { safeJsonParse } from '../utils/api'
import type { Item } from 'bdsa-react-components'
import type { CapturedRegion } from '../types'

interface UseImageSelectionProps {
  selectedItem: Item | null
  configLoaded: boolean
  dsaToken: string | null
  setSelectedItem: (item: Item | null) => void
  setHistogramData: (data: any) => void
  setHistogramLoading: (loading: boolean) => void
  setPpcData: (data: any) => void
  setIntensityMap: (map: any) => void
  setShowPpcLabel: (show: boolean) => void
  setCapturedRegions: Dispatch<SetStateAction<CapturedRegion[]>>
  setSavedParameterSets: (sets: any) => void
  setImageDimensions: (dims: { width: number, height: number } | null) => void
  setCurrentViewport: (viewport: any) => void
  autoDetectHue: (itemId: string) => Promise<void>
}

export function useImageSelection({
  selectedItem,
  configLoaded,
  dsaToken,
  setSelectedItem,
  setHistogramData,
  setHistogramLoading,
  setPpcData,
  setIntensityMap,
  setShowPpcLabel,
  setCapturedRegions,
  setSavedParameterSets,
  setImageDimensions,
  setCurrentViewport,
  autoDetectHue,
}: UseImageSelectionProps) {
  // Fetch image dimensions when item changes
  useEffect(() => {
    const itemId = (selectedItem as any)?._id || (selectedItem as any)?.id
    if (!itemId) {
      setImageDimensions(null)
      return
    }
    
    // Fetch image dimensions from backend
    fetch(`/api/images/${itemId}`)
      .then(async response => {
        if (response.ok) {
          return safeJsonParse(response)
        }
        throw new Error(`Failed to fetch image info: ${response.status}`)
      })
      .then((data: { width?: number, height?: number }) => {
        if (data.width && data.height) {
          setImageDimensions({ width: data.width, height: data.height })
        }
      })
      .catch(error => {
        console.error('Error fetching image dimensions:', error)
        setImageDimensions(null)
      })
  }, [selectedItem, setImageDimensions])

  // Handle thumbnail click - load histogram and auto-detect hue
  const handleThumbnailClick = useCallback(async (item: Item) => {
    console.log('Thumbnail clicked:', item)
    setSelectedItem(item)
    setHistogramData(null)
    setPpcData(null) // Clear PPC data when new image is selected
    setIntensityMap(null) // Clear intensity map when new image is selected
    setShowPpcLabel(false) // Hide label overlay when new image is selected
    setCapturedRegions([]) // Clear captured regions + region overlays when new image is selected
    // Hide any saved parameter set overlays when switching slides
    setSavedParameterSets((prev: any[]) => prev.map((s) => ({ ...s, visible: false })))
    setImageDimensions(null) // Clear image dimensions
    setCurrentViewport(null) // Clear viewport info
    setHistogramLoading(true)
    
    // Fetch histogram data
    const itemId = (item as any)._id || (item as any).id
    if (itemId) {
      try {
        const response = await fetch(`/api/ppc/histogram?item_id=${itemId}&width=1024&bins=256`)
        if (response.ok) {
          const data = await safeJsonParse(response)
          setHistogramData(data)
        } else {
          console.error('Failed to fetch histogram:', response.status)
        }
      } catch (error) {
        console.error('Error fetching histogram:', error)
      } finally {
        setHistogramLoading(false)
      }
      
      // Auto-detect hue parameters for new image
      await autoDetectHue(itemId)
    }
  }, [
    setSelectedItem,
    setHistogramData,
    setPpcData,
    setIntensityMap,
    setShowPpcLabel,
    setCapturedRegions,
    setSavedParameterSets,
    setImageDimensions,
    setCurrentViewport,
    setHistogramLoading,
    autoDetectHue,
  ])

  // Load histogram for saved item when config is loaded
  useEffect(() => {
    if (configLoaded && selectedItem && dsaToken) {
      const itemId = (selectedItem as any)._id || (selectedItem as any).id
      if (itemId) {
        console.log('🔄 Restoring saved item and loading histogram:', itemId)
        setHistogramLoading(true)
        fetch(`/api/ppc/histogram?item_id=${itemId}&width=1024&bins=256`)
          .then(async (histResponse) => {
            if (histResponse.ok) {
              const histData = await safeJsonParse(histResponse)
              setHistogramData(histData)
              console.log('✓ Histogram loaded for saved item')
            }
          })
          .catch((error) => {
            console.error('Failed to load histogram for saved item:', error)
          })
          .finally(() => {
            setHistogramLoading(false)
          })
        
        // Auto-detect hue parameters when restoring saved item
        autoDetectHue(itemId)
      }
    }
  }, [configLoaded, dsaToken, selectedItem, setHistogramData, setHistogramLoading, autoDetectHue])

  return { handleThumbnailClick }
}
