import React, { Component } from 'react';

const withErrorHandler = WrappedComponent => class extends Component {
    state = {
        error: null
    };

    render() {
        if (!this.state.error) {
            return <WrappedComponent {...this.props} />;
        }
        return (

        );
    }
};

export default withErrorHandler;