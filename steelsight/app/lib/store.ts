import { Inspection } from '../types/inspection';

let inspections: Inspection[] = [];

export const saveInspection = (inspection: Inspection) => {
  inspections = [inspection, ...inspections];
};

export const getInspections = () => {
  return [...inspections].sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
};

export const getInspection = (id: string) => {
  return inspections.find(i => i.id === id);
};

export const getStats = () => {
  const total = inspections.length;
  const defects = inspections.filter(i => i.status === 'REVIEW').length;
  const passed = total - defects;
  const defectRate = total ? Math.round((defects / total) * 100) : 0;
  return { total, defects, passed, defectRate };
};

export const generateId = () => {
  const num = inspections.length + 1;
  return `INS-${num.toString().padStart(5, '0')}`;
};
