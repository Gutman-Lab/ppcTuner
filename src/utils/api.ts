/**
 * API utility functions
 */

/**
 * Helper function to safely parse JSON responses (handles HTML error pages)
 */
export async function safeJsonParse(response: Response): Promise<any> {
  const contentType = response.headers.get('content-type')
  if (!contentType || !contentType.includes('application/json')) {
    // Response is not JSON, likely HTML error page
    // Clone response to read text without consuming original (though we'll throw anyway)
    const clonedResponse = response.clone()
    const text = await clonedResponse.text()
    const errorMsg = `Expected JSON but got ${contentType || 'unknown type'}. Status: ${response.status} ${response.statusText}`
    console.error('JSON parse error:', errorMsg)
    console.error('Response preview:', text.substring(0, 500))
    throw new Error(errorMsg)
  }
  try {
    return await response.json()
  } catch (error) {
    // Even if content-type says JSON, parsing might fail
    const clonedResponse = response.clone()
    const text = await clonedResponse.text()
    console.error('JSON parse failed despite JSON content-type. Response preview:', text.substring(0, 500))
    throw new Error(`Failed to parse JSON response: ${error}. Status: ${response.status}`)
  }
}

/**
 * Round a number to a specified number of decimal places
 */
export function round(value: number, decimals: number): number {
  return Math.round(value * Math.pow(10, decimals)) / Math.pow(10, decimals)
}
