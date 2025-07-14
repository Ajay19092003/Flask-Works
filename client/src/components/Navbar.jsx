import React from 'react'
import { Link } from 'react-router-dom'
import { LogOut, User, Heart, Truck, Building2, Stethoscope } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { useSocket } from '../context/SocketContext'

const Navbar = () => {
  const { user, logout } = useAuth()
  const { connected } = useSocket()

  const roleIcons = {
    victim: Heart,
    driver: Truck,
    hospital: Building2,
    medical_expert: Stethoscope
  }

  const roleLabels = {
    victim: 'Victim/Helper',
    driver: 'Ambulance Driver',
    hospital: 'Hospital Helpdesk',
    medical_expert: 'Medical Expert'
  }

  const RoleIcon = roleIcons[user?.role] || User

  return (
    <nav className="navbar">
      <div className="container">
        <div className="navbar-content">
          <div className="flex items-center gap-4">
            <Link to="/" className="navbar-brand">
              🚑 DisasterCare
            </Link>
            
            {/* Connection status indicator */}
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm text-gray-600">
                {connected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>

          <div className="navbar-nav">
            {user && (
              <>
                <div className="flex items-center gap-2 px-3 py-2 bg-gray-100 rounded-lg">
                  <RoleIcon size={16} />
                  <span className="text-sm font-medium">
                    {roleLabels[user.role]}
                  </span>
                </div>
                
                <div className="flex items-center gap-2 text-gray-700">
                  <User size={16} />
                  <span className="font-medium">{user.name}</span>
                </div>
                
                <button
                  onClick={logout}
                  className="btn btn-secondary"
                >
                  <LogOut size={16} />
                  Logout
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar