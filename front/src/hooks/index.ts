import { useStore } from '@nanostores/react';
import { stepperFormStore } from '@/store';
import { StepperFormData } from '@/types';

export const useStepperForm = () => {
  const store = useStore(stepperFormStore);

  const setStepperForm = (data: StepperFormData | undefined) => {
    stepperFormStore.set(data);
  };

  const resetStepperForm = () => {
    stepperFormStore.set(undefined);
  };

  return {
    stepperForm: store as StepperFormData,
    setStepperForm,
    resetStepperForm,
  };
};
