import { shallow, mount } from 'enzyme';
import ConnectedCompareModelVersionsView, {
  CompareModelVersionsView,
} from './CompareModelVersionsView';
import React from 'react';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import configureStore from 'redux-mock-store';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { RunInfo } from '../../experiment-tracking/sdk/MlflowMessages';

describe('unconnected tests', () => {
  let wrapper;
  let minimumProps;
  let commonProps;

  beforeEach(() => {
    minimumProps = {
      modelName: 'test',
      runsToVersions: { '123': 'dummy_version' },
      runUuids: ['123'],
      runInfos: [],
      metricLists: [],
      paramLists: [],
      runNames: [],
      runDisplayNames: [],
    };

    commonProps = {
      ...minimumProps,
      runInfos: [
        RunInfo.fromJs({
          run_uuid: '123',
          experiment_id: '0',
          user_id: 'test.user',
          status: 'FINISHED',
          start_time: '0',
          end_time: '21',
          artifact_uri: './mlruns',
          lifecycle_stage: 'active',
        }),
      ],
      metricLists: [['test_metric', 0.0, '321', '42']],
      paramLists: [['test_param', '0.0']],
    };
  });

  test('unconnected should render with minimal props without exploding', () => {
    wrapper = shallow(<CompareModelVersionsView {...minimumProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('check that the component renders correctly with common props', () => {
    wrapper = shallow(<CompareModelVersionsView {...commonProps} />);

    // Checking the breadcrumb renders correctly
    expect(
      wrapper.containsAllMatchingElements(['Registered Models', 'test', 'Comparing 1 Versions']),
    ).toEqual(true);

    // Checking the model version shows up
    expect(wrapper.containsAllMatchingElements(['Model Version:', 'dummy_version'])).toEqual(true);
  });
});

describe('connected tests', () => {
  let wrapper;
  let minimumProps;
  let minimalStore;
  let commonStore;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    minimumProps = {
      modelName: 'test',
      runsToVersions: { '123': 'dummy_version' },
    };

    minimalStore = mockStore({
      entities: {
        runInfosByUuid: { '123': RunInfo.fromJs({ dummy_key: 'dummy_value' }) },
        latestMetricsByRunUuid: { '123': 'dummy' },
        paramsByRunUuid: { '123': 'dummy' },
        tagsByRunUuid: { '123': 'dummy' },
      },
      apis: {},
    });

    commonStore = mockStore({
      entities: {
        runInfosByUuid: {
          '123': RunInfo.fromJs({
            run_uuid: '123',
            experiment_id: '0',
            user_id: 'test.user',
            status: 'FINISHED',
            start_time: '0',
            end_time: '21',
            artifact_uri: './mlruns',
            lifecycle_stage: 'active',
          }),
        },
        latestMetricsByRunUuid: {
          '123': { key: 'test_metric', value: 0.0, timestamp: '321', step: '42' },
        },
        paramsByRunUuid: { '123': { key: 'test_param', value: '0.0' } },
        tagsByRunUuid: { '123': { key: 'test_tag', value: 'test.user' } },
      },
      apis: {},
    });
  });

  test('connected should render with minimal props and minimal store without exploding', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ConnectedCompareModelVersionsView {...minimumProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find(ConnectedCompareModelVersionsView).length).toBe(1);
  });

  test('connected should render with minimal props and common store correctly', () => {
    wrapper = mount(
      <Provider store={commonStore}>
        <BrowserRouter>
          <ConnectedCompareModelVersionsView {...minimumProps} />
        </BrowserRouter>
      </Provider>,
    );

    expect(wrapper.find(ConnectedCompareModelVersionsView).length).toBe(1);

    // Checking the breadcrumb renders correctly
    expect(
      wrapper.containsAllMatchingElements(['Registered Models', 'test', 'Comparing 1 Versions']),
    ).toEqual(true);

    // Checking the model version shows up
    expect(wrapper.containsAllMatchingElements(['Model Version:', 'dummy_version'])).toEqual(true);
  });
});
