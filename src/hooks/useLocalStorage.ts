/**
 * Custom hook for localStorage persistence
 */
import { useEffect, useState } from 'react'
import type { Dispatch, SetStateAction } from 'react'

export function useLocalStorage<T>(key: string, initialValue: T): [T, Dispatch<SetStateAction<T>>] {
  // State to store our value
  // Pass initial state function to useState so logic is only executed once
  const [storedValue, setStoredValue] = useState<T>(() => {
    if (typeof window === 'undefined') {
      return initialValue
    }
    try {
      const item = window.localStorage.getItem(key)
      if (!item) {
        return initialValue
      }
      // Try to parse the item
      const parsed = JSON.parse(item)
      return parsed
    } catch (error) {
      // If parsing fails, the data is corrupted - clear it and return initial value
      console.error(`Error loading ${key} from localStorage (corrupted data):`, error)
      try {
        window.localStorage.removeItem(key)
        console.log(`Cleared corrupted ${key} from localStorage`)
      } catch (clearError) {
        console.error(`Failed to clear corrupted ${key} from localStorage:`, clearError)
      }
      return initialValue
    }
  })

  // Return a wrapped version of useState's setter function that
  // persists the new value to localStorage.
  const setValue: Dispatch<SetStateAction<T>> = (value) => {
    try {
      // Allow value to be a function so we have same API as useState
      const valueToStore = value instanceof Function ? value(storedValue) : value
      setStoredValue(valueToStore)
      if (typeof window !== 'undefined') {
        // Handle null/empty values - remove from localStorage if null or empty string
        if (valueToStore === null || valueToStore === '') {
          window.localStorage.removeItem(key)
        } else {
          window.localStorage.setItem(key, JSON.stringify(valueToStore))
        }
      }
    } catch (error) {
      console.error(`Error saving ${key} to localStorage:`, error)
    }
  }

  // Sync with localStorage changes from other tabs/windows
  useEffect(() => {
    if (typeof window === 'undefined') return

    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === key && e.newValue !== null) {
        try {
          const newValue = JSON.parse(e.newValue)
          setStoredValue(newValue)
        } catch (error) {
          console.error(`Error parsing storage change for ${key}:`, error)
        }
      } else if (e.key === key && e.newValue === null) {
        // Item was removed
        setStoredValue(initialValue)
      }
    }

    window.addEventListener('storage', handleStorageChange)
    return () => window.removeEventListener('storage', handleStorageChange)
  }, [key, initialValue])

  return [storedValue, setValue]
}
