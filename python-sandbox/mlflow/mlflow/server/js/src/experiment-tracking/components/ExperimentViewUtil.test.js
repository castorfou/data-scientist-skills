import React from 'react';
import { shallow, mount } from 'enzyme';
import ExperimentViewUtil, { TreeNode } from './ExperimentViewUtil';
import { BrowserRouter } from 'react-router-dom';

describe('ExperimentViewUtil', () => {
  test('getCheckboxForRow should render', () => {
    const component = ExperimentViewUtil.getCheckboxForRow(true, () => {}, 'div');
    const wrapper = shallow(component);
    expect(wrapper.length).toBe(1);
  });

  test('getRunInfoCellsForRow returns a row containing userid, start time, and status', () => {
    const runInfo = {
      user_id: 'user1',
      start_time: new Date('2020-01-02').getTime(),
      status: 'FINISHED',
    };
    const runInfoCells = ExperimentViewUtil.getRunInfoCellsForRow(
      runInfo,
      {},
      false,
      'div',
      () => {},
      [],
    );
    const renderedCells = runInfoCells.map((c) => mount(<BrowserRouter>{c}</BrowserRouter>));
    expect(renderedCells[0].find('.run-table-container').filter({ title: 'FINISHED' }).length).toBe(
      1,
    );
    const allText = renderedCells.map((c) => c.text()).join();
    expect(allText).toContain('user1');
    // The start_time is localized, so it may be anywhere from -12 to +14 hours, based on the
    // client's timezone.
    expect(
      allText.includes('2020-01-01') ||
        allText.includes('2020-01-02') ||
        allText.includes('2020-01-03'),
    ).toBeTruthy();
  });

  test('clicking on getRunMetadataHeaderCells sorts column if column is sortable', () => {
    const mockSortFn = jest.fn();
    const headerComponents = ExperimentViewUtil.getRunMetadataHeaderCells(
      mockSortFn,
      'user_id',
      true,
      'div',
      [],
    );
    // We assume that headerComponent[1] is the 'start_time' header
    const startTimeHeader = shallow(headerComponents[1]);
    startTimeHeader.find('.sortable').simulate('click');
    expect(mockSortFn.mock.calls[0][0]).toEqual(expect.stringContaining('start_time'));
    expect(mockSortFn.mock.calls[0][1]).toBeFalsy();
  });

  test('clicking on getRunMetadataHeaderCells does nothing if column is not sortable', () => {
    const mockSortFn = jest.fn();
    const headerComponents = ExperimentViewUtil.getRunMetadataHeaderCells(
      mockSortFn,
      'user_id',
      true,
      'div',
      [],
    );
    // We assume that headerComponent[0] is the 'status' header
    const statusHeader = shallow(headerComponents[0]);
    statusHeader.find('.run-table-container').simulate('click');
    expect(mockSortFn.mock.calls.length).toEqual(0);
  });

  test('getRunMetadataHeaderCells excludes excludedCols', () => {
    const headerComponents = ExperimentViewUtil.getRunMetadataHeaderCells(
      () => {},
      'user_id',
      true,
      'div',
      [ExperimentViewUtil.AttributeColumnLabels.DATE],
    );
    const headers = headerComponents.map((c) => shallow(c));
    headers.forEach((h) => {
      expect(h.text()).not.toContain(ExperimentViewUtil.AttributeColumnLabels.DATE);
    });

    // As a sanity check, let's make sure the headers contain some other column
    const userHeaders = headers.filter(
      (h) => h.text() === ExperimentViewUtil.AttributeColumnLabels.USER,
    );
    expect(userHeaders.length).toBe(1);
  });

  test('computeMetricRanges returns the correct min and max value for a metric', () => {
    const metrics = [
      { key: 'foo', value: 1 },
      { key: 'foo', value: 2 },
      { key: 'foo', value: 0 },
    ];
    const metricsByRun = [metrics];
    const ranges = ExperimentViewUtil.computeMetricRanges(metricsByRun);
    expect(ranges.foo.min).toBe(0);
    expect(ranges.foo.max).toBe(2);
  });

  test('TreeNode finds the correct root', () => {
    const root = new TreeNode('root');
    const child = new TreeNode('child');
    child.parent = root;
    const grandchild = new TreeNode('grandchild');
    grandchild.parent = child;

    expect(grandchild.findRoot().value).toBe('root');
  });

  test('TreeNode knows which node is the root', () => {
    const root = new TreeNode('root');
    const child = new TreeNode('child');
    child.parent = root;

    expect(root.isRoot()).toBeTruthy();
    expect(child.isRoot()).toBeFalsy();
  });

  test('TreeNode detects a cycle', () => {
    const root = new TreeNode('root');
    const child = new TreeNode('child');
    child.parent = root;
    const child2 = new TreeNode('child2');
    child2.parent = root;
    const grandchild = new TreeNode('grandchild');
    grandchild.parent = child2;
    root.parent = grandchild;

    expect(grandchild.isCycle()).toBeTruthy();
  });
});
