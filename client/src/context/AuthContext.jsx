import React, { createContext, useContext, useState, useEffect } from 'react'
import axios from 'axios'
import toast from 'react-hot-toast'

const AuthContext = createContext()

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

// Configure axios defaults
const API_URL = 'http://localhost:3001/api'
axios.defaults.baseURL = API_URL

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  // Set up axios interceptor for token
  useEffect(() => {
    const token = localStorage.getItem('token')
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
    }

    // Add response interceptor to handle token expiration
    const interceptor = axios.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401 || error.response?.status === 403) {
          logout()
          toast.error('Session expired. Please login again.')
        }
        return Promise.reject(error)
      }
    )

    return () => {
      axios.interceptors.response.eject(interceptor)
    }
  }, [])

  // Check for existing token on app load
  useEffect(() => {
    const token = localStorage.getItem('token')
    const savedUser = localStorage.getItem('user')
    
    if (token && savedUser) {
      try {
        const userData = JSON.parse(savedUser)
        setUser(userData)
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
      } catch (error) {
        console.error('Error parsing saved user data:', error)
        localStorage.removeItem('token')
        localStorage.removeItem('user')
      }
    }
    
    setLoading(false)
  }, [])

  const login = async (email, password) => {
    try {
      setLoading(true)
      const response = await axios.post('/auth/login', { email, password })
      
      const { token, user: userData } = response.data
      
      // Save token and user data
      localStorage.setItem('token', token)
      localStorage.setItem('user', JSON.stringify(userData))
      
      // Set axios header
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
      
      // Update state
      setUser(userData)
      
      toast.success(`Welcome back, ${userData.name}!`)
      return { success: true, user: userData }
      
    } catch (error) {
      const message = error.response?.data?.error || 'Login failed'
      toast.error(message)
      return { success: false, error: message }
    } finally {
      setLoading(false)
    }
  }

  const register = async (formData) => {
    try {
      setLoading(true)
      const response = await axios.post('/auth/register', formData)
      
      const { token, user: userData } = response.data
      
      // Save token and user data
      localStorage.setItem('token', token)
      localStorage.setItem('user', JSON.stringify(userData))
      
      // Set axios header
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
      
      // Update state
      setUser(userData)
      
      toast.success(`Registration successful! Welcome, ${userData.name}!`)
      return { success: true, user: userData }
      
    } catch (error) {
      const message = error.response?.data?.error || 'Registration failed'
      toast.error(message)
      return { success: false, error: message }
    } finally {
      setLoading(false)
    }
  }

  const logout = () => {
    // Clear storage
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    
    // Clear axios header
    delete axios.defaults.headers.common['Authorization']
    
    // Clear state
    setUser(null)
    
    toast.success('Logged out successfully')
  }

  const updateUser = (userData) => {
    setUser(userData)
    localStorage.setItem('user', JSON.stringify(userData))
  }

  const value = {
    user,
    loading,
    login,
    register,
    logout,
    updateUser
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}