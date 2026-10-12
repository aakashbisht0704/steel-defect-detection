import { Severity, InspectionStatus } from '../types/inspection';

export function calculateSeverity(confidence: number, defectClass: string): { severity: Severity, status: InspectionStatus } {
  if (confidence < 0.5 || defectClass === 'none') {
    return { severity: 'LOW', status: 'PASS' };
  }
  let severity: Severity = 'LOW';
  if (confidence < 0.8) severity = 'LOW';
  else if (confidence < 0.95) severity = 'MEDIUM';
  else severity = 'HIGH';
  return { severity, status: 'REVIEW' };
}
