import React from 'react';
import { shallow } from 'enzyme';
import { ModelStageTransitionDropdown } from './ModelStageTransitionDropdown';
import { Stages } from '../constants';
import { Dropdown } from 'antd';

describe('ModelStageTransitionDropdown', () => {
  let wrapper;
  let minimalProps;
  let commonProps;

  beforeEach(() => {
    minimalProps = {
      currentStage: Stages.NONE,
    };
    commonProps = {
      ...minimalProps,
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ModelStageTransitionDropdown {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should omit current stage in dropdown', () => {
    const props = {
      ...minimalProps,
      currentStage: Stages.STAGING,
    };
    wrapper = shallow(<ModelStageTransitionDropdown {...props} />);
    wrapper.find('.stage-transition-dropdown').simulate('click');
    const menuHtml = shallow(wrapper.find(Dropdown).props().overlay).html();
    expect(menuHtml).not.toContain(Stages.STAGING);
    expect(menuHtml).toContain(Stages.PRODUCTION);
    expect(menuHtml).toContain(Stages.NONE);
    expect(menuHtml).toContain(Stages.ARCHIVED);
  });

  test('handleMenuItemClick', () => {
    const mockOnSelect = jest.fn();
    const props = {
      ...commonProps,
      onSelect: mockOnSelect,
    };
    const activity = {};
    wrapper = shallow(<ModelStageTransitionDropdown {...props} />);
    const instance = wrapper.instance();
    instance.handleMenuItemClick(activity);
    instance.state.handleConfirm();
    expect(mockOnSelect).toHaveBeenCalledWith(activity);
  });
});
