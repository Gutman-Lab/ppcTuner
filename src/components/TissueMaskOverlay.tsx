import { useState, useEffect } from 'react'

// Tissue mask overlay component - now uses backend-generated mask
export function TissueMaskOverlay({ itemId, backgroundThreshold }: { itemId: string, backgroundThreshold: number }) {
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
  }, [itemId, backgroundThreshold, maskImage])
  
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
