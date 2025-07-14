import React, { createContext, useContext, useEffect, useState } from 'react'
import io from 'socket.io-client'
import toast from 'react-hot-toast'
import { useAuth } from './AuthContext'

const SocketContext = createContext()

export const useSocket = () => {
  const context = useContext(SocketContext)
  if (!context) {
    throw new Error('useSocket must be used within a SocketProvider')
  }
  return context
}

export const SocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null)
  const [connected, setConnected] = useState(false)
  const { user } = useAuth()

  useEffect(() => {
    if (user) {
      // Initialize socket connection
      const newSocket = io('http://localhost:3001')
      
      newSocket.on('connect', () => {
        console.log('Connected to server:', newSocket.id)
        setConnected(true)
        
        // Join user-specific room
        newSocket.emit('join_room', { 
          room: `user_${user.id}`,
          userId: user.id,
          userRole: user.role 
        })
      })

      newSocket.on('disconnect', () => {
        console.log('Disconnected from server')
        setConnected(false)
      })

      // Emergency-related events
      newSocket.on('new_emergency_assignment', (data) => {
        if (user.role === 'driver') {
          toast.success('🚨 New Emergency Assignment!', {
            duration: 6000,
            icon: '🚑',
            style: {
              background: '#ff6b6b',
              color: 'white',
            },
          })
          
          // Play notification sound (if available)
          try {
            const audio = new Audio('/notification.mp3')
            audio.play().catch(e => console.log('Audio play failed:', e))
          } catch (e) {
            console.log('Audio not available')
          }
        }
      })

      newSocket.on('driver_accepted', (data) => {
        if (user.role === 'victim') {
          toast.success('🚑 Driver accepted your request!', {
            duration: 5000,
            icon: '✅',
          })
        }
      })

      newSocket.on('driver_location_update', (data) => {
        // Handle real-time driver location updates
        console.log('Driver location update:', data)
      })

      newSocket.on('medical_guidance', (data) => {
        if (user.role === 'driver') {
          toast.success('📋 Medical guidance received', {
            duration: 8000,
            icon: '🩺',
          })
        }
      })

      // Hospital alert events
      newSocket.on('hospital_alert_1', (data) => {
        if (user.role === 'hospital') {
          toast.success('🚨 Ambulance en route to your hospital!', {
            duration: 10000,
            icon: '🏥',
            style: {
              background: '#00d2ff',
              color: 'white',
            },
          })
          
          // Play emergency sound
          try {
            const audio = new Audio('/hospital-alert.mp3')
            audio.play().catch(e => console.log('Audio play failed:', e))
          } catch (e) {
            console.log('Audio not available')
          }
        }
      })

      newSocket.on('hospital_alert_2', (data) => {
        if (user.role === 'hospital') {
          toast.warning('⚠️ Ambulance is 200m away!', {
            duration: 8000,
            icon: '📍',
            style: {
              background: '#ffa500',
              color: 'white',
            },
          })
        }
      })

      newSocket.on('hospital_alert_3', (data) => {
        if (user.role === 'hospital') {
          toast.success('✅ Ambulance has arrived!', {
            duration: 6000,
            icon: '🏁',
            style: {
              background: '#28a745',
              color: 'white',
            },
          })
        }
      })

      // Connection error handling
      newSocket.on('connect_error', (error) => {
        console.error('Socket connection error:', error)
        setConnected(false)
      })

      setSocket(newSocket)

      // Cleanup on unmount
      return () => {
        newSocket.close()
        setSocket(null)
        setConnected(false)
      }
    }
  }, [user])

  const joinRoom = (roomName) => {
    if (socket) {
      socket.emit('join_room', { room: roomName })
    }
  }

  const leaveRoom = (roomName) => {
    if (socket) {
      socket.emit('leave_room', { room: roomName })
    }
  }

  const updateLocation = (lat, lng, emergencyId, driverId) => {
    if (socket) {
      socket.emit('update_location', {
        lat,
        lng,
        emergencyId,
        driverId,
        timestamp: new Date()
      })
    }
  }

  const sendMessage = (room, message) => {
    if (socket) {
      socket.emit('send_message', {
        room,
        message,
        sender: user?.id,
        senderName: user?.name,
        timestamp: new Date()
      })
    }
  }

  const value = {
    socket,
    connected,
    joinRoom,
    leaveRoom,
    updateLocation,
    sendMessage
  }

  return (
    <SocketContext.Provider value={value}>
      {children}
    </SocketContext.Provider>
  )
}