import React from 'react';
import { shallow } from 'enzyme';
import { MetricsPlotControls, X_AXIS_RELATIVE } from './MetricsPlotControls';
import { CHART_TYPE_BAR, CHART_TYPE_LINE } from './MetricsPlotPanel';

describe('unit tests', () => {
  let wrapper;
  const minimalPropsForLineChart = {
    distinctMetricKeys: ['metric_0', 'metric_1'],
    selectedMetricKeys: ['metric_0'],
    selectedXAxis: X_AXIS_RELATIVE,
    handleXAxisChange: jest.fn(),
    handleShowPointChange: jest.fn(),
    handleMetricsSelectChange: jest.fn(),
    handleYAxisLogScaleChange: jest.fn(),
    handleLineSmoothChange: jest.fn(),
    chartType: CHART_TYPE_LINE,
    initialLineSmoothness: 1,
    yAxisLogScale: true,
    xAxis: X_AXIS_RELATIVE,
    onLayoutChange: jest.fn(),
    showPoint: false,
  };
  const minimalPropsForBarChart = { ...minimalPropsForLineChart, chartType: CHART_TYPE_BAR };

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<MetricsPlotControls {...minimalPropsForLineChart} />);
    expect(wrapper.length).toBe(1);
    wrapper = shallow(<MetricsPlotControls {...minimalPropsForBarChart} />);
    expect(wrapper.length).toBe(1);
  });

  test('should show x-axis controls for line chart', () => {
    wrapper = shallow(<MetricsPlotControls {...minimalPropsForLineChart} />);
    expect(wrapper.find('.show-point-toggle')).toHaveLength(1);
    expect(wrapper.find('.smoothness-toggle')).toHaveLength(1);
    expect(wrapper.find('.x-axis-radio')).toHaveLength(3);
  });

  test('should not show x-axis controls for bar chart', () => {
    wrapper = shallow(<MetricsPlotControls {...minimalPropsForBarChart} />);
    expect(wrapper.find('.show-point-toggle')).toHaveLength(0);
    expect(wrapper.find('.smoothness-toggle')).toHaveLength(0);
    expect(wrapper.find('.x-axis-radio')).toHaveLength(0);
  });
});
