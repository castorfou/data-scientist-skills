import React from 'react';
import { RequestStateWrapper } from '../../common/components/RequestStateWrapper';
import { ErrorCodes } from '../../common/constants';
import { shallow } from 'enzyme';
import { Spinner } from '../../common/components/Spinner';
import { ErrorWrapper } from '../../common/utils/ActionUtils';

const activeRequest = {
  id: 'a',
  active: true,
};

const completeRequest = {
  id: 'a',
  active: false,
  data: { run_id: 'run_id' },
};

const errorRequest = {
  id: 'a',
  active: false,
  error: new ErrorWrapper({
    responseText: `{"error_code": "${ErrorCodes.RESOURCE_DOES_NOT_EXIST}"}`,
  }),
};

test('Renders loading page when requests are not complete', () => {
  const wrapper = shallow(
    <RequestStateWrapper requests={[activeRequest, completeRequest]}>
      <div>I am the child</div>
    </RequestStateWrapper>,
  );
  expect(wrapper.find(Spinner)).toHaveLength(1);
});

test('Renders children when requests are complete', () => {
  const wrapper = shallow(
    <RequestStateWrapper requests={[completeRequest]}>
      <div className='child'>I am the child</div>
    </RequestStateWrapper>,
  );
  expect(wrapper.find('div.child')).toHaveLength(1);
  expect(wrapper.find('div.child').text()).toContain('I am the child');
});

test('Throws exception if child is a React element and wrapper has bad request.', () => {
  try {
    shallow(
      <RequestStateWrapper requests={[errorRequest]}>
        <div className='child'>I am the child</div>
      </RequestStateWrapper>,
    );
  } catch (e) {
    expect(e.message).toContain('GOTO error boundary');
  }
});

test('Throws exception if errorRenderFunc returns undefined and wrapper has bad request.', () => {
  try {
    shallow(
      <RequestStateWrapper
        requests={[errorRequest]}
        errorRenderFunc={() => {
          return undefined;
        }}
      >
        <div className='child'>I am the child</div>
      </RequestStateWrapper>,
    );
    assert.fail();
  } catch (e) {
    expect(e.message).toContain('GOTO error boundary');
  }
});

test('Render func works if wrapper has bad request.', () => {
  const wrapper = shallow(
    <RequestStateWrapper requests={[activeRequest, completeRequest, errorRequest]}>
      {(isLoading, shouldRenderError, requests) => {
        if (shouldRenderError) {
          expect(requests).toEqual([activeRequest, completeRequest, errorRequest]);
          return <div className='error'>Error!</div>;
        }
        return <div className='child'>I am the child</div>;
      }}
    </RequestStateWrapper>,
  );
  expect(wrapper.find('div.error')).toHaveLength(1);
  expect(wrapper.find('div.error').text()).toContain('Error!');
});
