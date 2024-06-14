import { useStore } from '@nanostores/react';
import { useEffect, useState } from 'react';
import { resultStore, stepperFormStore } from '@/store';
import { ApiResponse, StepperFormData } from '@/types';

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

export const useWidth = () => {
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);

  useEffect(() => {
    const windowWidthHandler = () => {
      setWindowWidth(window.innerWidth);
    };

    window.addEventListener('resize', windowWidthHandler);

    return () => {
      window.removeEventListener('resize', windowWidthHandler);
    };
  }, []);

  return windowWidth;
};

export const useResult = () => {
  const store = useStore(resultStore);

  const setResult = (data: ApiResponse | undefined) => {
    resultStore.set(data);
  };

  const resetResult = () => {
    stepperFormStore.set(undefined);
  };

  return {
    store,
    setResult,
    resetResult,
  };
};
