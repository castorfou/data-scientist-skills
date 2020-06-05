import React from 'react';
import { shallow } from 'enzyme';
import ShowArtifactPdfView from './ShowArtifactPdfView';

describe('ShowArtifactPdfView', () => {
  let wrapper;
  let instance;
  let minimalProps;
  let commonProps;

  beforeEach(() => {
    minimalProps = {
      path: 'fakepath',
      runUuid: 'fakeUuid',
    };
    // Mock the `getArtifact` function to avoid spurious network errors
    // during testing
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve('some content');
    });
    commonProps = { ...minimalProps, getArtifact };
    wrapper = shallow(<ShowArtifactPdfView {...commonProps} />);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ShowArtifactPdfView {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render loading text when view is loading', () => {
    instance = wrapper.instance();
    instance.setState({ loading: true });
    expect(wrapper.find('.artifact-pdf-view-loading').length).toBe(1);
  });

  test('should render error message when error occurs', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.reject(new Error('my error text'));
    });
    const props = { ...minimalProps, getArtifact: getArtifact };
    wrapper = shallow(<ShowArtifactPdfView {...props} />);
    setImmediate(() => {
      expect(wrapper.find('.artifact-pdf-view-error').length).toBe(1);
      expect(wrapper.instance().state.loading).toBe(false);
      expect(wrapper.instance().state.html).toBeUndefined();
      expect(wrapper.instance().state.error).toBeDefined();
      done();
    });
  });

  test('should render PDF in container', () => {
    wrapper.setState({ loading: false });
    wrapper.setProps({ path: 'fake.pdf', runUuid: 'fakeRunId' });
    expect(wrapper.find('.pdf-outer-container')).toHaveLength(1);
    expect(wrapper.find('.pdf-viewer')).toHaveLength(1);
    expect(wrapper.find('.paginator')).toHaveLength(1);
    expect(wrapper.find('.document')).toHaveLength(1);
  });

  test('should call fetchPdf on component update', () => {
    instance = wrapper.instance();
    instance.fetchPdf = jest.fn();
    wrapper.setProps({ path: 'newpath', runUuid: 'newRunId' });
    expect(instance.fetchPdf).toHaveBeenCalledTimes(1);
  });
});
