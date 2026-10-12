export type DefectClass = 'crazing' | 'inclusion' | 'patches' | 'pitted_surface' | 'rolled_in_scale' | 'scratches'

export type Severity = 'LOW' | 'MEDIUM' | 'HIGH'
export type InspectionStatus = 'PASS' | 'REVIEW'

export type Inspection = {
  id: string
  filename: string
  prediction: {
    class: DefectClass | 'none'
    confidence: number
    probabilities: Record<DefectClass, number>
  }
  severity: Severity
  status: InspectionStatus
  model: { name: string; version: string }
  images: { original: string; gradcam: string; gradcam_overlay: string }
  attentionCoverage: number
  createdAt: string
}
