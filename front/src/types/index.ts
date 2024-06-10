import { BackgroundFormData } from '@/components';

export type StepperFormData = {
  productPicture?: { fileData: string; fileName: string; mimeType: string };
} & BackgroundFormData;
