import React, { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import io from 'socket.io-client'

// Context
import { AuthProvider, useAuth } from './context/AuthContext'
import { SocketProvider } from './context/SocketContext'

// Components
import Navbar from './components/Navbar'
import LoadingSpinner from './components/LoadingSpinner'

// Pages
import LandingPage from './pages/LandingPage'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'

// User Role Dashboards
import VictimDashboard from './pages/VictimDashboard'
import DriverDashboard from './pages/DriverDashboard'
import HospitalDashboard from './pages/HospitalDashboard'
import MedicalExpertDashboard from './pages/MedicalExpertDashboard'

// Protected Route Component
function ProtectedRoute({ children, allowedRoles }) {
  const { user, loading } = useAuth()
  
  if (loading) {
    return <LoadingSpinner />
  }
  
  if (!user) {
    return <Navigate to="/login" />
  }
  
  if (allowedRoles && !allowedRoles.includes(user.role)) {
    return <Navigate to="/" />
  }
  
  return children
}

// Role-based redirect component
function RoleBasedRedirect() {
  const { user } = useAuth()
  
  if (!user) {
    return <Navigate to="/login" />
  }
  
  switch (user.role) {
    case 'victim':
      return <Navigate to="/victim/dashboard" />
    case 'driver':
      return <Navigate to="/driver/dashboard" />
    case 'hospital':
      return <Navigate to="/hospital/dashboard" />
    case 'medical_expert':
      return <Navigate to="/expert/dashboard" />
    default:
      return <Navigate to="/login" />
  }
}

function AppContent() {
  const { user } = useAuth()

  return (
    <div className="min-h-screen">
      {user && <Navbar />}
      
      <Routes>
        {/* Public Routes */}
        <Route path="/" element={user ? <RoleBasedRedirect /> : <LandingPage />} />
        <Route path="/login" element={user ? <RoleBasedRedirect /> : <LoginPage />} />
        <Route path="/register" element={user ? <RoleBasedRedirect /> : <RegisterPage />} />
        
        {/* Protected Routes */}
        <Route 
          path="/victim/dashboard" 
          element={
            <ProtectedRoute allowedRoles={['victim']}>
              <VictimDashboard />
            </ProtectedRoute>
          } 
        />
        
        <Route 
          path="/driver/dashboard" 
          element={
            <ProtectedRoute allowedRoles={['driver']}>
              <DriverDashboard />
            </ProtectedRoute>
          } 
        />
        
        <Route 
          path="/hospital/dashboard" 
          element={
            <ProtectedRoute allowedRoles={['hospital']}>
              <HospitalDashboard />
            </ProtectedRoute>
          } 
        />
        
        <Route 
          path="/expert/dashboard" 
          element={
            <ProtectedRoute allowedRoles={['medical_expert']}>
              <MedicalExpertDashboard />
            </ProtectedRoute>
          } 
        />
        
        {/* Catch all route */}
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
      
      {/* Toast notifications */}
      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
        }}
      />
    </div>
  )
}

function App() {
  return (
    <Router>
      <AuthProvider>
        <SocketProvider>
          <AppContent />
        </SocketProvider>
      </AuthProvider>
    </Router>
  )
}

export default App