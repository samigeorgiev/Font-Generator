import React, { Component } from 'react';

import styles from './index.module.css';

class Home extends Component {
    render() {
        return (
            <main className={styles.Home}>
                <h1 className={styles.PageHeading}>Font Generator</h1>
                <p className={styles.AppDescription}>Generate novel font combinations with neural networks</p>
            </main>
        );
    }
}

export default Home;