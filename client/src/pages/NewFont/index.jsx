import React, { Component } from 'react';

import Slider from 'components/Slider';

import styles from './index.module.css';

class NewFont extends Component {
    state = {
        value: 5
    }

    render() {
        return (
            <main className={styles.NewFont}>
                <Slider change={e => {console.log(e.target.value); this.setState({value: e.target.value})}}/>
            </main>
        );
    }
}

export default NewFont;