import { useNavigate } from 'react-router-dom';
import { Actions, BackgroundForm, Stepper, BackgroundFormData } from '@/components';
import { useStepperForm } from '@/hooks';

export const BackgroundPage = () => {
  const navigate = useNavigate();
  const { stepperForm, setStepperForm } = useStepperForm();
  const onSubmit = (data: BackgroundFormData) => {
    setStepperForm({ ...stepperForm, ...data });
    navigate('/parameters');
  };
  const getBack = () => navigate('/');
  return (
    <Stepper>
      <BackgroundForm onSubmit={onSubmit} actions={<Actions getBack={getBack} />} />
    </Stepper>
  );
};
