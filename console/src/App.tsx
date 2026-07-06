import { useState } from 'react'
import DoctorPicker from './pages/DoctorPicker'
import PatientList from './pages/PatientList'
import PatientChart from './pages/PatientChart'
import { clearToken } from './api/client'

type Screen = 'picker' | 'list' | 'chart'

export default function App() {
  const [screen,    setScreen]    = useState<Screen>('picker')
  const [doctorInfo, setDoctorInfo] = useState<{ userId: string; roleLabel: string; userName: string } | null>(null)
  const [patientId, setPatientId] = useState<string | null>(null)

  const handleDoctorSelect = (userId: string, roleLabel: string, userName: string) => {
    setDoctorInfo({ userId, roleLabel, userName })
    setPatientId(null)
    setScreen('list')
  }

  const handlePatientSelect = (id: string) => {
    setPatientId(id)
    setScreen('chart')
  }

  const handleBackToList = () => {
    setPatientId(null)
    setScreen('list')
  }

  const handleBackToPicker = () => {
    clearToken()
    setDoctorInfo(null)
    setPatientId(null)
    setScreen('picker')
  }

  if (screen === 'picker') {
    return <DoctorPicker onSelect={handleDoctorSelect} />
  }

  if (screen === 'list') {
    return (
      <PatientList
        doctorInfo={doctorInfo}
        onSelect={handlePatientSelect}
        onBack={handleBackToPicker}
      />
    )
  }

  return (
    <PatientChart
      patientId={patientId!}
      onBack={handleBackToList}
    />
  )
}
