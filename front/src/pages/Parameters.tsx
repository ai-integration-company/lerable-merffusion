import { useNavigate } from 'react-router-dom';
import { Actions, ParametersForm, Stepper } from '@/components';

export const ParametersPage = () => {
  const navigate = useNavigate();
  const onSubmit = () => {
    navigate('/results');
  };
  return (
    <Stepper>
      <ParametersForm onSubmit={onSubmit} actions={<Actions backButtonDisabled />} />
    </Stepper>
  );
};
