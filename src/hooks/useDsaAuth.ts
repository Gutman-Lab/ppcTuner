/**
 * Custom hook for DSA authentication and API configuration
 */
import { useState, useEffect, useCallback, useMemo } from 'react'
import type { AppConfig } from '../types'
import { safeJsonParse } from '../utils/api'

export function useDsaAuth() {
  const [apiBaseUrl, setApiBaseUrl] = useState<string>('http://bdsa.pathology.emory.edu:8080/api/v1')
  const [dsaToken, setDsaToken] = useState<string | null>(null)
  const [configLoaded, setConfigLoaded] = useState(false)
  const [startFolder, setStartFolder] = useState<string>('')
  const [startFolderType, setStartFolderType] = useState<string>('collection')

  // Load configuration from backend
  useEffect(() => {
    async function loadConfig() {
      try {
        const response = await fetch('/api/config')
        if (response.ok) {
          const config: AppConfig = await safeJsonParse(response)
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
                  const userData = await safeJsonParse(res)
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
                          const data = await safeJsonParse(collRes)
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
          }
        }
      } catch (error) {
        console.error('Failed to load config:', error)
      } finally {
        setConfigLoaded(true)
      }
    }
    loadConfig()
  }, [])

  // Memoize fetchFn for ThumbnailGrid to prevent unnecessary re-fetches
  const thumbnailGridFetchFn = useCallback(
    async (url: string, options?: RequestInit) => {
      // Always add token if available, even if it's in apiHeaders
      // This ensures authentication works even if apiHeaders isn't used
      const headers: Record<string, string> = {
        ...(options?.headers as Record<string, string> || {}),
      }

      if (dsaToken) {
        headers['Girder-Token'] = dsaToken
      }

      console.log('ThumbnailGrid fetch:', url)
      console.log('ThumbnailGrid has token:', !!dsaToken)

      const response = await fetch(url, {
        ...options,
        headers,
      })

      console.log('ThumbnailGrid response status:', response.status, 'for', url)

      if (!response.ok && response.status === 401) {
        console.error('ThumbnailGrid 401 Unauthorized - token may be invalid or expired')
        const errorText = await response.clone().text()
        console.error('Error response:', errorText.substring(0, 200))
      }

      // Clone the response for debugging without consuming the original
      if (response.ok) {
        const clonedResponse = response.clone()
        try {
          const data = await safeJsonParse(clonedResponse)
          console.log('ThumbnailGrid response data:', data)
          console.log('Number of items:', Array.isArray(data) ? data.length : 'Not an array')
        } catch (e) {
          console.log('Could not parse response for logging:', e)
        }
      } else {
        const clonedResponse = response.clone()
        try {
          const text = await clonedResponse.text()
          console.error('ThumbnailGrid error response:', response.status, text.substring(0, 200))
        } catch (e) {
          console.error('ThumbnailGrid error status:', response.status)
        }
      }

      // Return the original, unconsumed response
      return response
    },
    [dsaToken]
  )

  // Memoize fetchFn for FolderBrowser to prevent unnecessary re-fetches
  const folderBrowserFetchFn = useCallback(
    async (url: string, options?: RequestInit) => {
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
    },
    [dsaToken]
  )

  // Memoize apiHeaders to prevent ThumbnailGrid from re-rendering unnecessarily
  const thumbnailGridApiHeaders = useMemo(() => {
    return dsaToken ? { 'Girder-Token': dsaToken } : undefined
  }, [dsaToken])

  // Memoize apiHeaders for FolderBrowser to prevent unnecessary re-renders
  const folderBrowserApiHeaders = useMemo(() => {
    return dsaToken ? { 'Girder-Token': dsaToken } : undefined
  }, [dsaToken])

  return {
    apiBaseUrl,
    dsaToken,
    configLoaded,
    startFolder,
    startFolderType,
    thumbnailGridFetchFn,
    folderBrowserFetchFn,
    thumbnailGridApiHeaders,
    folderBrowserApiHeaders,
  }
}
